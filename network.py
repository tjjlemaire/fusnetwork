# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-13 13:37:40
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-12-03 16:53:01

import itertools
from tqdm import tqdm
from neuron import h
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from logger import logger
from utils import *


def get_NetStim(freq=1, start=None, number=1e9, noise=0):
    '''
    Wrapper around NetStim allowing to set parameters in 1-liner.
    
    :param freq: spiking frequency of pre-synaptic drive (Hz)
    :param start (optional): start time (ms)
    :param number (optional): total number of pre-synaptic spikes
    :return: NetStim object
    '''
    # If start time not provided, set it to random value inside first cycle
    if start is None:
        if noise == 0:
            start = np.random.uniform(0, 1 / freq * 1e3)  # ms
        else:
            start = 0
    
    # Log process 
    fstr = f'{freq:.1f}'
    if noise > 0:
        fstr += f' ± {noise * freq:.1f}'
    s = f'creating {fstr} Hz pre-synaptic drive starting at {start:.1f} ms'
    if number < 1e9:
        stop = start + number / freq * 1e3
        s += f'and stopping at {stop:.1f} ms'
    logger.info(s)

    # Create NetStim object
    stim = h.NetStim()

    # Set NetStim parameters
    stim.number = number
    stim.start = start # ms
    stim.interval = 1e3 / freq  # ms
    stim.noise = noise

    # Return NetStim object
    return stim


def get_NetStim_params(stim):
    ''' Extract dictionary of NetStim parameters. '''
    return {
        'freq': 1e3 / stim.interval,  # Hz
        'start': stim.start,  # ms
        'number': stim.number,  # number of spikes
        'noise': stim.noise,  # noise factor
    }


class NeuralNetwork:
    ''' Interface class to a network of neurons. '''

    # Conversion factors
    UM_TO_CM = 1e-4
    NA_TO_MA = 1e-6
    S_TO_US = 1e6
    S_TO_MS = 1e3

    # Default model parameters
    Acell = 11.84e3  # Cell membrane area (um2)
    G_RS_RS = 0.002  # Synaptic weight between RS cells, from Plaksin 2016 (uS)
    g_RS_RS = (G_RS_RS / S_TO_US) / (Acell * UM_TO_CM**2)  # Synaptic weight between RS cells (S/cm2)
    
    mechname = 'RS'  # NEURON mechanism name
    vrest = -71.9  # neurons resting potential (mV)

    TIME_KEY = 'time (ms)'
    STIM_KEY = 'stim'
    NODE_KEY = 'node'
    REP_KEY = 'rep'
    RESP_KEY = 'ΔFR/FR'
    ISPPA_KEY = 'Isppa (W/cm2)'
    DUR_KEY = 'dur (ms)'

    # Dictionary of variables to record during simulation (besides time), by category
    record_dict = {
        'generic': {
            'T': 'temperature (°C)',
            'v': 'membrane potential (mV)',
        },
        'conductances': {
            'gNabar': 'sodium conductance (uS/cm2)',
            'gKdbar': 'potassium conductance (uS/cm2)',
            'gMbar': 'slow non-inactivating potassium conductance (uS/cm2)',
            'gLeak': 'leak conductance (uS/cm2)',
            'gKT': 'thermally-driven Potassium conductance (uS/cm2)',
            'gNaKPump': 'sodium-potassium pump conductance (uS/cm2)',
        },
        'currents': {
            'iDrive': 'driving current (mA/cm2)',
            'iStim': 'stimulus-driven current (mA/cm2)',
            'iNa': 'sodium current (mA/cm2)',
            'iKd': 'potassium current (mA/cm2)',
            'iM': 'slow non-inactivating potassium current (mA/cm2)',
            'iLeak': 'leak current (mA/cm2)',
            'iKT': 'thermally-driven Potassium current (uA/cm2)',
            'iNaKPump': 'sodium-potassium pump current (mA/cm2)',
        },
    }

    TIMEVAR_SUFFIX = '_t'  # suffix to add to time-varying conductance variables

    @classmethod
    def record_keys(cls, kind):
        ''' 
        Return serialized list of recordable variables corresponding to one or more categories.

        :param kind: category name or list of category names
        '''
        # If single category name provided, convert to list
        if isinstance(kind, str):
            kind = [kind]
        
        # Return list of recordable variables
        l = []
        for k in kind:
            try:
                keys = list(cls.record_dict[k].keys())
            except KeyError:
                raise ValueError(f'Invalid recordable variable category: {k}')
            l = l + keys
        return l
    
    @classmethod
    def parse_record_keys(cls, keys):
        '''
        Parse a list of recordable variable names and/or recordable categories into
        a list of recordable variables

        :param keys: list of recordable variable names and/or recordable categories
        :return: list of recordable variable names
        '''
        # If no keys provided, return empty list
        if keys is None:
            return []

        # If single key provided, convert to list
        if isinstance(keys, str):
            keys = [keys]
        
        # Parse list of keys
        l = []
        # For each key
        for k in keys:
            # If recording category, add keys of corresponding category
            if k in cls.record_dict:
                l = l + cls.record_keys(k)
            # Otherwise, look for matching recording variable and add it
            else:
                found = False
                for rdict in cls.record_dict.values():
                    if k in rdict.keys():
                        l.append(k)
                        found = True
                        break

                # If no match found, raise error
                if not found:
                    raise ValueError(f'Invalid recordable variable: {k}')
        
        # Remove duplicates and return list
        return list(dict.fromkeys(l))

    @classmethod
    def filter_record_keys(cls, include=None, exclude=None):
        ''' 
        Construct a list of recordable variables given inclusion and exclustion criteria
        
        :param include: list of variables to include
        :param exclude: list of variables to exclude
        :return: filtered record dictionary
        '''
        # Get serialized list of all recordable variables
        allkeys = cls.record_keys(list(cls.record_dict.keys()))

        # List variables that should be excluded
        toremove = []
        if exclude is not None:
            exclude = cls.parse_record_keys(exclude)
            for k in allkeys:
                if k in exclude:
                    toremove.append(k)
        if include is not None:
            include = cls.parse_record_keys(include)
            for k in allkeys:
                if k not in include:
                    toremove.append(k)
        toremove = list(set(toremove))

        # Return filtered list
        allkeys = [k for k in allkeys if k not in toremove]
        return allkeys

    def __init__(self, nnodes, connect=True, synweight=None, verbose=True, **kwargs):
        '''
        Initialize a neural network with a given number of nodes.
        
        :param nnodes: number of nodes
        :param connect: whether to connect nodes (default: True)
        :param synweight: synaptic weight (uS) (default: None)
        :param verbose: verbosity flag (default: True)
        :param kwargs: model parameters passed as "key=value" pairs (optional)
        '''
        # Set verbosity
        self.verbose = verbose
        # Create nodes
        self.create_nodes(nnodes)
        # Set biophysics
        self.set_biophysics()
        # Initialize empty containers for NEURON objects
        self.init_obj_lists()
        # Connect nodes with appropriate synaptic weights, if requested
        if connect and nnodes > 1:
            conkwargs = {}
            if synweight is not None:
                conkwargs['weight'] = synweight
            self.connect_nodes(**conkwargs)
        # Set model parameters, if provided
        self.set_mech_param(**kwargs)
        # Set default stimulus parameters and simulation duration to None
        self.start = None
        self.dur = None
        self.tstop = None
        # Log initialization completion
        self.log('initialized')
        
    def __repr__(self):
        ''' String representation. '''
        return f'{self.__class__.__name__}({self.size})'

    def log(self, msg, warn=False):
        ''' Log message with verbose-dependent logging level. '''
        if warn:
            logfunc = logger.warning
        else:
            logfunc = logger.info if self.verbose else logger.debug
        logfunc(f'{self}: {msg}')
    
    def get_common_section_dim(self, A):
        ''' 
        Compute common diameter and length of a cylindrical section 
        to produce a specific membrane area along the cylinder surface.
        
        :param A: target membrane area (um2)
        :return: common diameter and length measure (um)
        '''
        # Surface area along cylinder length: A = πdL
        # Assuming d = L, we have: A = πd^2, therefore: d = L = sqrt(A/π)
        return np.sqrt(A / np.pi)  # um
    
    def create_node(self, name):
        '''
        Create node section and adjust its dimensions to
        obtain a target membrane area

        :param name: node name
        :return: node section
        '''
        x = self.get_common_section_dim(self.Acell)
        sec = h.Section(name=name)
        sec.diam = x
        sec.L = x
        return sec
    
    def create_nodes(self, nnodes):
        ''' Create a given number of nodes. '''
        self.nodes = [self.create_node(f'node{i}') for i in range(nnodes)]
        self.log(f'created {self.size} node{"s" if self.size > 1 else ""}')
        if self.refarea != self.Acell:
            raise ValueError(
                f'resulting node membrane area ({self.refarea:.2f} um2) differs from reference value ({self.Acell:.2f} um2)')
    
    def init_obj_lists(self):
        ''' Initialize containers to store NEURON objects for the model. '''
        self.drive_objs = [None] * self.size
        self.syn_objs = None
        self.netcon_objs = None
    
    @property
    def size(self):
        return len(self.nodes)

    @property
    def nodelist(self):
        ''' Return list of node names. '''
        return [node.name() for node in self.nodes]

    @property
    def refarea(self):
        ''' Surface area of first node in the model (um2). '''
        return self.nodes[0](0.5).area()  # um2
    
    def set_biophysics(self):
        ''' Assign membrane mechanisms to all nodes. '''
        for node in self.nodes:
            node.insert(self.mechname)
    
    def get_mech_param(self, key, inode=0):
        ''' 
        Get a specific mechanism parameter from a given node.
        
        :param key: parameter name
        :param inode: node index (default: 0)
        '''
        try:
            return getattr(self.nodes[inode], f'{key}_{self.mechname}')
        except AttributeError:
            raise ValueError(f'"{self.mechname}" mechanism does not have parameter "{key}"')
        
    def __getattr__(self, name):
        '''
        Redefinition of __getattr__ method to allow to get mechanism parameters
        
        :param name: parameter name
        '''
        # If name is a mechanism parameter, return it
        try:
            return self.get_mech_param(name)
        # Otherwise, raise error
        except ValueError:
            raise AttributeError(f'"{self}" does not have "{name}" attribute')
    
    def _set_mech_param(self, key, value, inode=None):
        '''
        Set a specific mechanism parameter on a specific set of nodes.
        
        :param key: parameter name
        :param value: parameter value
        :param inode: node index (default: None, i.e. all nodes)
        '''
        # Get default parameter value
        default_val = self.get_mech_param(key)

        # If value no different than default, return
        if value == default_val:
            return

        # Identify target nodes
        if inode is None:
            inode = list(range(self.size))
        elif isinstance(inode, int):
            inode = [inode]
        
        # Set new parameter value on nodes
        if len(inode) == 1:
            nodestr = f'node {inode[0]}'
        elif len(inode) == self.size:
            nodestr = 'all nodes'
        else:
            nodestr = f'nodes {inode}'
        self.log(f'setting {key} = {value} on {nodestr}', warn=True)
        for i in inode:
            setattr(self.nodes[i], f'{key}_{self.mechname}', value)
    
    def set_mech_param(self, inode=None, **kwargs):
        ''' 
        Wrapper around "_set_mech_param" method allowing to set multiple mechanism parameters
        at once using "key=value" syntax.
        
        :param inode: node index (default: None, i.e. all nodes)
        :param kwargs: dictionary of parameter names and values
        '''
        for k, v in kwargs.items():
            self._set_mech_param(k, v, inode=inode)
    
    def to_absolute_conductance(self, g):
        '''
        Convert synaptic weight from relative to absolute conductance.

        :param g: synaptic weight (S/cm2)
        :return: synaptic weight (uS)
        '''
        # Multiply by neuron soma membrane area, and convert to uS
        return (g * self.S_TO_US) * (self.Acell * self.UM_TO_CM**2) 
    
    def to_relative_conductance(self, G):
        '''
        Convert synaptic weight from absolute to relative conductance

        :param G: synaptic weight (uS)
        :return: synaptic weight (S/cm2)
        '''
        # Divide by neuron soma membrane area, and convert to S/cm2
        return (G / self.S_TO_US) / (self.Acell * self.UM_TO_CM**2)

    def to_current_density(self, I):
        '''
        Convert current to current density for compatibility with NEURON formalism.

        :param I: absolute current (nA)
        :return: current density (mA/cm2)
        '''
        return I * self.NA_TO_MA / (self.Acell * self.UM_TO_CM**2)
    
    def set_presyn_drive(self, inode=None, delay=0, weight=4.5e-4, sync=False, **kwargs):
        ''' 
        Set a presynaptic drive on a specific set of nodes.

        :param inode: node index (default: None, i.e. all nodes)
        :param delay: synaptic delay between pre-synaptic drive and target nodes (ms)
        :param weight: synaptic weight, i.e. maximal synaptic conductance (S/cm2)
        :param sync: whether to synchronize pre-synaptic drive across nodes (default: False)
        :param kwargs: dictionary of parameter names and values fed to "get_NetStim" function
        '''
        # Identify target nodes
        if inode is None:
            inode = list(range(self.size))
        elif isinstance(inode, int):
            inode = [inode]
        
        self.log(f'setting pre-synaptic drive on nodes {inode}')

        # If sync flag enabled, generate unique pre-synaptic drive with passed parameters
        if sync:
            stim = get_NetStim(**kwargs)
        else:
            stim = None

        # For each target node
        for i in inode:
            if self.is_drive_set(inode=i):
                logger.warning(f'modifying pre-synaptic drive on node {i}')
            # Create simple synapse object and attach it to node 
            syn = h.ExpSyn(self.nodes[i](0.5))
            syn.tau = 1  # decay time constant (ms)

            # If sync flag disabled, generate node-specific pre-synaptic drive with passed parameters
            if not sync:
                stim = get_NetStim(**kwargs)

            # Create network connection between pre-synaptic drive and target node
            nc = h.NetCon(stim, syn)
            nc.weight[0] = self.to_absolute_conductance(weight) 
            nc.delay = delay

            # Append netstim, synapse and netcon objects to network class atributes 
            self.drive_objs[i] = (stim, syn, nc)
    
    def get_drive_params(self, inode=None):
        '''
        Extract pre-synaptic drive parameters from a specific set of nodes.

        :param inode: node index (default: None, i.e. all nodes)
        :return: dictionary of pre-synaptic drive parameters
        '''
        # Identify target nodes
        if inode is None:
            inode = list(range(self.size))
        elif isinstance(inode, int):
            inode = [inode]

        # Get drive parameters
        params = {}
        for i in inode:
            if not self.is_drive_set(inode=i):
                raise ValueError(f'cannot extract drive parameters for node {i}: no pre-synaptic drive set')
            params[f'node{i}'] = get_NetStim_params(self.drive_objs[i][0])
        
        # If only one node, return dictionary of parameters
        if len(params) == 1:
            return list(params.values())[0]
        
        # If multiple nodes, construct dataframe of parameters, check that they are 
        # identical across nodes, and return summary dictionary
        params = pd.DataFrame(params).T
        if any(params.nunique(axis=0) != 1):
            raise ValueError('pre-synaptic drive parameters differ across nodes')
        return params.iloc[0].to_dict()

    def is_drive_set(self, inode=None):
        ''' 
        Check if presynaptic drive is set on a specific set of nodes

        :param inode: node index (default: None, i.e. all nodes)
        :return: boolean(s) indicating whether drive is set on specified node(s) 
        '''
        # Identify target nodes
        if inode is None:
            inode = list(range(self.size))
        elif isinstance(inode, int):
            inode = [inode]
        
        # Check status of pre-synaptic drive in all nodes
        out = [self.drive_objs[i] is not None for i in inode]
        if len(out) == 1:
            out = out[0]

        # Return
        return out

    def remove_presyn_drive(self, inode=None):
        ''' 
        Remove presynaptic drive on a specific set of nodes

        :param inode: node index (default: None, i.e. all nodes)
        '''
        # Identify target nodes
        if inode is None:
            inode = list(range(self.size))
        elif isinstance(inode, int):
            inode = [inode]

        # For each target node
        for i in inode:
            if not self.is_drive_set(inode=i):
                logger.warning(f'no pre-synaptic drive to remove at node {i}')
            else:
                stim, _, nc = self.drive_objs[i]
                stim.number = 0
                nc.weight[0] = 0
                self.drive_objs[i] = None
     
    def connect(self, ipresyn, ipostsyn, weight=None):
        '''
        Connect a source node to a target node with a specific synapse model
        and synaptic weight.

        :param ipresyn: index of the pre-synaptic node
        :param ipostsyn: index of the post-synaptic node
        :param weight: synaptic weight (S/cm2). If None, use default value.
        '''
        # If synaptic weight not provided, use default RS-RS value
        if weight is None:
            weight = self.g_RS_RS
        
        # Create bi-exponential AMPA synapse and attach it to target node
        syn = h.Exp2Syn(self.nodes[ipostsyn](0.5))
        syn.tau1 = 0.1  # rise time constant (ms)
        syn.tau2 = 3.0  # decay time constant (ms)
        
        # Generate network-connection between pre and post synaptic nodes
        nc = h.NetCon(
            self.get_var_ref(ipresyn, 'v'),  # trigger variable: pre-synaptic voltage (mV)
            syn,  # synapse object (already attached to post-synaptic node)
            sec=self.nodes[ipresyn]  # pre-synaptic node
        )

        # Assign netcon attributes
        nc.threshold = 0.  # pre-synaptic voltage threshold (mV)
        nc.delay = 1.  # synaptic delay (ms)
        nc.weight[0] = self.to_absolute_conductance(weight)  # synaptic weight (uS)

        # Append synapse and netcon objects to network class atributes 
        self.syn_objs.append(syn)
        self.netcon_objs.append(nc)
    
    def connect_nodes(self, **kwargs):
        ''' Form all specific connections between network nodes '''
        self.log('connecting all node pairs')
        self.syn_objs = []
        self.netcon_objs = []
        for pair in itertools.combinations(range(self.size), 2):
            self.connect(*pair, **kwargs)
            self.connect(*pair[::-1], **kwargs)
    
    def is_connected(self):
        ''' Return whether network nodes are connected. '''
        return hasattr(self, 'netcon_objs') and self.netcon_objs is not None
    
    def disconnect_nodes(self):
        ''' Clear all synapses and network connection objects between nodes '''
        logger.info(f'removing all {len(self.netcon_objs)} connections between nodes')
        self.syn_objs = None
        self.netcon_objs = None
    
    def set_synaptic_weight(self, w):
        ''' Set synaptic weight on all connections. '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        self.log(f'setting all synaptic weights to {w:.2e} S/cm2')
        for nc in self.netcon_objs:
            nc.weight[0] = self.to_absolute_conductance(w)  # synaptic weight (uS)
    
    def get_synaptic_weight(self):
        ''' Return synaptic weight on all connections. '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        return self.to_relative_conductance(self.netcon_objs[0].weight[0])
 
    def get_vector_list(self):
        ''' Return model-sized list of NEURON vectors. '''
        return [h.Vector() for node in self.nodes]
    
    def get_var_ref(self, inode, varname):
        ''' 
        Get reference to specific variable on a given node.

        :param inode: node index
        :param varname: variable name
        '''
        # If variable is iDrive, return reference to synapse current
        if varname == 'iDrive':
            if self.drive_objs[inode] is None:
                raise ValueError(f'cannot record drive current on node {inode}: no drive set')
            return self.drive_objs[inode][1]._ref_i
        
        # Get full reference key to variable
        if varname == 'v':
            varkey = '_ref_v'
        else:
            varkey = f'_ref_{varname}_{self.mechname}'
        
        # Return reference on appropriate node
        return getattr(self.nodes[inode](0.5), varkey)

    def record_time(self, key='t'):
        ''' Record time vector during simulation '''
        self.probes[key] = h.Vector()
        self.probes[key].record(h._ref_t)
    
    def record_on_all_nodes(self, varname):
        '''
        Record (a) given variable(s) on all nodes.

        :param varname: variable name(s)
        '''
        # Recursively call function if multiple variables are provided
        if isinstance(varname, (tuple, list)):
            for v in varname:
                self.record_on_all_nodes(v)
            return
        
        # Strip trailing suffix if present (for conductances)
        # to define variable key
        varkey = varname.rstrip(self.TIMEVAR_SUFFIX)

        # Initialize recording probes list for variable
        self.probes[varkey] = self.get_vector_list()

        # Record variable on all nodes
        logger.debug(f'recording "{varname}" on all nodes with key "{varkey}"')
        for inode in range(self.size):
            self.probes[varkey][inode].record(self.get_var_ref(inode, varname))
    
    def get_disabled_currents(self):
        ''' 
        List disabled membrane currents in the model (i.e. those with null
        reference conductances)

        :return: list of (conductance key, current key) tuples for all disabled currents
        '''
        l = []
        for gkey in self.record_dict['conductances'].keys():
            if self.get_mech_param(gkey) == 0:
                l.append((
                    gkey,   # conductance key
                    f'i{gkey[1:]}'.rstrip('bar')   # current key
                ))
        self.log(f'disabled currents: {", ".join([ikey for _, ikey in l])}')
        is_drive_set = self.is_drive_set()
        if isinstance(is_drive_set, list):
            is_drive_set = all(is_drive_set)
        if not is_drive_set:
            l.append(('iDrive',))
        if not self.is_stim_set():
            l.append(('iStim',))
        return l
    
    def get_disabled_keys(self):
        ''' List all keys associated to disabled currents '''
        return list(itertools.chain(*self.get_disabled_currents()))
    
    def set_recording_probes(self):
        ''' Initialize recording probes for all nodes. '''
        # Initialize probes dictionary
        self.probes = {}

        # Assign time probe
        self.record_time()

        disabled_keys = self.get_disabled_keys()

        # For other variables of interest
        for rectype, recdict in self.record_dict.items():
            # Get list of keys to record
            reckeys = list(recdict.keys())

            # Remove keys corresponding to disabled currents from list of keys to record
            reckeys = [k for k in reckeys if k not in disabled_keys]

            # For conductances, append suffix to key to record
            # conductance variable instead of its reference value
            if rectype == 'conductances':
                reckeys = [f'{k}{self.TIMEVAR_SUFFIX}' for k in reckeys]

            # Record variables on all nodes
            self.record_on_all_nodes(reckeys)
    
    def extract_from_recording_probes(self):
        '''
        Extract output vectors from recording probes.
        
        :return: multi-indexed dataframe storing the time course of all recording variables across nodes
        '''
        # Extract time vector
        tprobe = self.probes.pop('t')
        t = pd.Series(np.array(tprobe.to_python()), name=self.TIME_KEY)  # ms

        # Create output dataframe 
        dfout = pd.DataFrame()

        for k, v in self.probes.items():
            # Extract 2D array of variable time course across nodes
            vpernode = np.array([x.to_python() for x in v])

            # If probe recorded absolute current, convert to current density
            if k == 'iDrive':
                vpernode = self.to_current_density(vpernode)
            
            # Create multi-indexed series from 2D array, and append to output dataframe
            dfout[k] = pd.DataFrame(vpernode.T, columns=self.nodelist, index=t).T.stack()

        # Return output dataframe
        dfout.index.names = [self.NODE_KEY, self.TIME_KEY]
        return dfout

    @classmethod
    def extract_time(cls, data):
        '''
        Extract time vector from simulation output dataframe

        :param data: multi-indexed simulation output dataframe
        :return: 1D time vector (ms)
        '''
        if isinstance(data.index, pd.MultiIndex):
            itime = data.index.names.index(cls.TIME_KEY)
            return data.index.levels[itime].values  # ms
        else:
            if data.index.name != cls.TIME_KEY:
                raise ValueError(f'cannot extract time: data index = {data.index.name}')
            return data.index.values
    
    @classmethod
    def extract_var(cls, data, key):
        '''
        Extract variable time course across nodes from simulation output dataframe
        
        :param data: multi-indexed simulation output dataframe
        :param key: variable name
        :return: 2D array of variable time course across nodes
        '''
        return data[key].unstack().values

    def get_stim_waveform(self, start, dur, amp):
        ''' 
        Define a stimulus waveform as a vector of time/amplitude pairs, 
        based on global stimulus parameters.

        :param start: stimulus start time (ms)
        :param dur: stimulus duration (ms)
        :param amp: stimulus amplitude
        :return: (time - amplitude) waveform vector
        '''
        return np.array([
            [0, 0],  # start at 0
            [start, 0],  # hold at 0 until stimulus onset
            [start, amp],  # switch to amplitude
            [start + dur, amp],  # hold at amplitude until stimulus offset
            [start + dur, 0],  # switch back to 0
            [start + dur + 10, 0],  # hold at 0 for 10 ms
        ])
    
    def vecstr(self, values, err=None, prefix=None, suffix=None, detailed=True):
        ''' 
        Return formatted string representation of node-specific values
        
        :param values: values per node
        :param err (optional): value error per node
        :param prefix (optional): prefix string
        :param suffix (optional): suffix string
        :param detailed: whether a detailed log is requested
        :return: descriptive string for the vector
        '''
        if isinstance(values, pd.DataFrame):
            err = values['err'].values
            values = values['mean'].values

        # Format input as iterable if not already
        if not isinstance(values, (tuple, list, np.ndarray, pd.Series)):
            values = [values]
        
        # Determine logging precision based on input type
        precision = 1 if isinstance(values[0], float) else 0

        # Format values as strings
        l = [f'{x:.{precision}f}' if x is not None else 'N/A' for x in values]
        if err is not None:
            errl = [f'±{x:.{precision}f}' if x is not None else '' for x in err]
            l = [f'{x} {err}' for x, err in zip(l, errl)]

        # Detailed mode: add node index to each item, and format as itemized list
        if detailed:
            if prefix is not None:
                l = [f'{prefix} {item}' for item in l]
            l = [f'    - node {i}: {x}' for i, x in enumerate(l)]
            if suffix is not None:
                l = [f'{item} {suffix}' for item in l]
            return '\n'.join(l)
        
        # Non-detailed mode: format as comma-separated list
        else:
            s = ', '.join(l)
            if len(l) > 1:
                s = f'[{s}]'
            if prefix is not None:
                s = f'{prefix} {s}'
            if suffix is not None:
                s = f'{s} {suffix}'
            return s

    def set_stim(self, amps, start=None, dur=None):
        ''' Set stimulus per node node with specific waveform parameters. '''
        # If stimulus start/duration provided, update class attribute
        if start is not None:
            self.start = start
        if dur is not None:
            self.dur = dur

        # Check that stimulus start and duration are valid
        if self.start is None:
            raise ValueError('No stimulus start time defined')
        elif self.start < 0:
            raise ValueError('Stimulus start time must be positive')
        if self.dur is None:
            raise ValueError('No stimulus duration defined')
        elif self.dur <= 0:
            raise ValueError('Stimulus duration must be strictly positive')

        # If scalar amplitude provided, expand to all nodes
        if isinstance(amps, (int, float)):
            amps = [amps] * self.size

        # Check that stimulus amplitudes are valid
        if len(amps) != self.size:
            raise ValueError(f'Number of stimulus amplitudes ({len(amps)}) does not match number of nodes {self.size}')
        if any(amp < 0 for amp in amps):
            raise ValueError('Stimulus amplitude must be positive')
        
        amps_str = self.vecstr(amps, suffix='W/cm2')
        self.log(f'setting {self.dur:.2f} ms stimulus with node-specific amplitudes:\n{amps_str}')
        
        # Set stimulus vectors
        self.h_yvecs = []
        for amp in amps:
            tvec, yvec = self.get_stim_waveform(self.start, self.dur, amp).T
            self.h_yvecs.append(h.Vector(yvec))
        self.h_tvec = h.Vector(tvec)

        # Play stimulus on all nodes with node-specific amplitudes
        for inode in range(self.size):
            self.h_yvecs[inode].play(
                self.get_var_ref(inode, 'I'), self.h_tvec, True)
    
    def is_stim_set(self):
        ''' Return whether stimulus is set. '''
        return hasattr(self, 'h_tvec') and hasattr(self, 'h_yvecs')
    
    def get_stim_vecs(self):
        ''' Return stimulus time-course and amplitude vectors. '''
        tvec = np.array(self.h_tvec.to_python())
        stimvecs = np.array([y.to_python() for y in self.h_yvecs])
        return tvec, stimvecs
    
    def interpolate_stim_vec(self, tstim, xstim, teval):
        ''' 
        Interpolate stimulus waveform along a set of evaluation times
        
        :param tref: stimulus time vector (ms)
        :param xstim: stimulus waveform vector
        :param teval: evaluation time vector (ms)
        '''
        return interp1d(
            tstim, xstim, 
            kind='previous', 
            bounds_error=False,
            fill_value=(0., xstim[-1])
        )(teval)
    
    def interpolate_stim_data(self, teval):
        '''
        Interpolate stimulus waveform along simulation time vector

        :param teval: evaluation time vector (ms)
        :return: multi-indexed dataframe storing the interpolated time course
            of the stimulus across nodes
        '''
        tvec, stimvecs = self.get_stim_vecs()
        stimdata = pd.DataFrame(index=teval)
        for node, xstim in zip(self.nodelist, stimvecs):
            stimdata[node] = self.interpolate_stim_vec(tvec, xstim, teval)
        return stimdata.T.stack().rename(self.STIM_KEY)
    
    def remove_stim(self):
        ''' Remove stimulus from all nodes. '''
        if not self.is_stim_set():
            self.log('no stimulus to remove')
        self.log('removing stimulus')
        for inode in range(self.size):
            self.h_yvecs[inode].play_remove()
        del self.h_tvec
        del self.h_yvecs
        self.start = None
        self.dur = None
        
    def simulate(self, tstop=None, nreps=1):
        '''
        Run a simulation for a given duration.
        
        :param tstop: simulation duration (ms)
        :param nreps: number of simulation repetitions (default: 1)
        :return: (time vector, voltage-per-node array) tuple
        '''
        # If multiple repetitions requested, recursively call function
        if nreps > 1:
            ireps = range(nreps)
            data = []
            for irep in ireps:
                self.log(f'repetition {irep + 1}/{nreps}')
                data.append(self.simulate(tstop=tstop))
            return pd.concat(data, axis=0, keys=ireps, names=[self.REP_KEY])
        
        # If simulation duration provided, update class attribute
        if tstop is not None:
            self.tstop = tstop
        
        # If no simulation duration defined, raise error
        if self.tstop is None:
            raise ValueError('No simulation duration defined')
        
        # Check that simulation duration outlasts stimulus waveform
        if self.is_stim_set() and self.tstop < self.h_tvec.x[-1]:
            raise ValueError('Simulation duration must be longer than stimulus waveform offset')
        
        # Initialize recording probes
        self.set_recording_probes()

        # Run simulation
        self.log(f'simulating for {self.tstop:.2f} ms')
        h.finitialize(self.vrest)
        while h.t < self.tstop:
            h.fadvance()
        
        # Convert NEURON vectors to numpy arrays
        self.log('extracting output results')
        data = self.extract_from_recording_probes()

        # If stimulus is set, interpolate stimulus waveforms along simulation time vector
        if self.is_stim_set():
            data[self.STIM_KEY] = self.interpolate_stim_data(self.extract_time(data))

        # Compute and log max temperature increase per node
        ΔT = self.compute_metric(data, 'ΔT')
        self.log(f'max temperature increase:\n{self.vecstr(ΔT, prefix="ΔT =", suffix="°C")}')
        
        # Count number of spikes and average firing rate per node
        nspikes = self.compute_metric(data, 'nspikes')
        self.log(f'number of spikes:\n{self.vecstr(nspikes, prefix="n =", suffix="spikes")}')
        FRs = self.compute_metric(data, 'FR')
        self.log(f'firing rate:\n{self.vecstr(FRs, prefix="FR =", suffix="Hz")}')

        # If stimulus is set, compute stimulus-evoked response
        if self.is_stim_set():
            stimresp = self.compute_metric(data, self.RESP_KEY)
            self.log(f'stimulus response:\n{self.vecstr(stimresp * 100, prefix=f"{self.RESP_KEY} =", suffix="%")}')
        
        # Return time and dictionary arrays of recorded variables
        return data
    
    def extract_ap_times(self, t, v, vref=0):
        '''
        Extract action potential times from a given voltage trace. 
        
        :param t: time vector (ms)
        :param v: voltage vector (mV), or array of voltage vectors
        :param vref: reference voltage (mV)
        :return: action potential times (ms), or array of action potential times
        '''
        # Recursively call function if voltage vectors array is provided
        if v.ndim > 1:
            return [self.extract_ap_times(t, vv) for vv in v]
    
        # Find indexes where where voltage crossed threshold with positive slope
        i_aps = np.where(np.diff(np.sign(v - vref)) > 0)[0]

        # Return corresponding action potential times
        return t[i_aps]

    def extract_ap_counts(self, t, vpernode):
        ''' Extract spike counts per node from a given array of voltage traces. '''
        return np.array([len(self.extract_ap_times(t, v)) for v in vpernode])
    
    def extract_firing_rates(self, t, vpernode):
        ''' 
        Extract firing rates mean and standard deviation per node from a given array of voltage traces.
        
        :param t: time vector (ms)
        :param vpernode: array of voltage vectors
        :return: vectors of mean and standard deviation of firing rates per node (Hz)
        '''
        # Extract list of spikes timings per node
        tspikes_per_node = self.extract_ap_times(t, vpernode)
        FRs = []

        # For each node
        for tspikes in tspikes_per_node:
            # If more than 1 spike detected, compute firing rate
            if len(tspikes) > 1:
                FR = 1 / np.diff(tspikes) * self.S_TO_MS  # Hz
            # Otherwise, set firing rate to None
            else:
                FR = None
            FRs.append(FR)
        
        # For each node, compute mean and standard deviation of firing rate
        mu_FRs = [FR.mean() if FR is not None else None for FR in FRs]  # Hz
        sigma_FRs = [FR.std() if FR is not None else None for FR in FRs]  # Hz

        # Return vectors of mean and standard deviation of firing rates per node
        return mu_FRs, sigma_FRs
    
    def extract_stim_response(self, data):
        '''
        Extract stimulus-evoked response from simulation output, as the relative 
        change in firing rate between the STIM-ON window and the rest of the 
        simulation duration.

        :param data: multi-indexed simulation output dataframe
        :return: array of stimulus response per node 
        '''
        # Check that stimulus is set
        if not self.is_stim_set():
            raise ValueError('cannot extract evoked response: stimulus not set')
        
        # Extract stimulus time bounds
        stimbounds = self.start, self.start + self.dur

        # Extract list of spikes timings per node
        tspikes_per_node = self.extract_ap_times(
            self.extract_time(data), self.extract_var(data, 'v'))

        # For each node
        stimresps = []
        for tspikes in tspikes_per_node:
            # Convert spikes list to numpy array
            tspikes = np.asarray(tspikes)
            # Classify spikes as during our outside stimulus window
            is_stim_on = np.logical_and(tspikes >= stimbounds[0], tspikes <= stimbounds[1])
            # Compute firing rate during stimulus window
            FR_stim = is_stim_on.sum() / self.dur * 1000  # Hz
            # Compute firing rate outside stimulus window
            FR_base = (tspikes.size - is_stim_on.sum()) / (self.tstop - self.dur) * 1000  # Hz
            # If no spikes detected outside window, set stimulus response to None
            if FR_base == 0:
                stimresp = None
            # Otherwise, compute stimulus response as stim-evoked relative change in firing rate
            else:
                stimresp = (FR_stim - FR_base) / FR_base
            # Append stimulus response to list
            stimresps.append(stimresp)
        
        # Return list of stimulus response per node
        return stimresps

    def compute_metric(self, data, metric):
        '''
        Compute metric across nodes from simulation output.
        
        :param data: multi-indexed simulation output dataframe
        :param metric: metric name
        :return: array(s) of metric value per node
        '''
        # If input data has extra dimensions, group by these extra dims
        # and compute metric across groups
        gby = [k for k in data.index.names if k not in (self.NODE_KEY, self.TIME_KEY)]
        if len(gby) > 0:
            if len(gby) == 1:
                gby = gby[0]
            mdict = {}
            for gkey, gdata in data.groupby(gby):
                mdict[gkey] = self.compute_metric(gdata.droplevel(gby), metric)
            return pd.concat(mdict.values(), axis=0, keys=mdict.keys(), names=as_iterable(gby))

        # Extract time vecotr from data
        t = self.extract_time(data) # ms

        # Extract metric from data
        if metric == 'nspikes':
            m = self.extract_ap_counts(t, self.extract_var(data, 'v'))
        elif metric == 'ΔT':
            T = self.extract_var(data, 'T')  # °C
            m = np.max(T, axis=1) - np.min(T, axis=1)
        elif metric == 'FR':
            m = self.extract_firing_rates(t, self.extract_var(data, 'v'))
        elif metric == self.RESP_KEY:
            m = self.extract_stim_response(data)
        else:
            raise ValueError(f'Invalid metric: {metric}')

        # If metric is a tuple, convert to mean - err dataframe  
        if isinstance(m, tuple):
            m = pd.DataFrame({'mean': m[0], 'err': m[1]})
        # Otherwise, convert to series
        else:
            m = pd.Series(m, name=metric)
        
        # Set index
        m.index = self.nodelist
        m.index.name = self.NODE_KEY

        # Return
        return m
    
    def filter_output(self, data, **kwargs):
        '''
        Filter simulation output data according to inclusion and exclusion criteria.

        :param data: multi-indexed simulation output dataframe
        :param kwargs: inclusion and exclusion criteria
        :return: filtered output dataframe
        '''
        # Extract list of keys from inclusion and exclusion criteria
        keys = self.filter_record_keys(**kwargs)

        # If stimulus is set, add stimulus key to list of keys
        if self.is_stim_set():
            keys.append(self.STIM_KEY)

        # Get intersection between keys and output data keys 
        # (in case some currents are disabled)
        keys = [k for k in keys if k in data.keys()]

        # Return filtered output dataframe
        return data[keys]

    def plot_results(self, data, tref='onset', gmode='abs', addstimspan=True, title=None, **kwargs):
        '''
        Plot the time course of variables recorded during simulation.
        
        :param data: multi-indexed simulation output dataframe
        :param tref: time reference for x-axis (default: 'onset')
        :param gmode: conductance plotting mode (default: 'abs'). One of:
            - "abs" (for absolute)
            - "rel" (for relative)
            - "norm" (for normalized)
            - "log" (for logarithmic) 
        :param addstimspan: whether to add a stimulus span on all axes (default: True)
        :param title: optional figure title (default: None)
        :return: figure handle 
        '''
        # Check that plotting mode is valid
        if gmode not in ['abs', 'rel', 'log', 'norm']:
            raise ValueError(f'Invalid conductance plotting mode: {gmode}')

        # Filter output 
        data = self.filter_output(data, **kwargs)

        # Log
        self.log('plotting results')

        # Assess which variable types are present in data
        hasstim = self.STIM_KEY in data
        hastemps = 'T' in data
        hasconds = 'conductances' in self.record_dict and any(k in data for k in self.record_dict['conductances'])
        hascurrs = 'currents' in self.record_dict and any(k in data for k in self.record_dict['currents'])
        hasvoltages = 'v' in data

        # Create figure with appropriate number of rows and columns
        hrow = 1.5
        wcol = 3.5
        nrows = int(hasstim) + int(hastemps) + int(hasconds) + int(hascurrs) + int(hasvoltages)
        nnodes_out = len(data.index.unique(self.NODE_KEY))
        assert nnodes_out == self.size, f'number of nodes ({self.size}) does not match number of output voltage traces ({nnodes_out})'
        ncols = self.size
        fig, axes = plt.subplots(nrows, ncols, figsize=(wcol * ncols + 1.5, hrow * nrows), sharex=True, sharey='row')
        if ncols == 1:
            axes = np.atleast_2d(axes).T
        sns.despine(fig=fig)
        for ax in axes[-1]:
            ax.set_xlabel(self.TIME_KEY)
        for inode, ax in enumerate(axes[0]):
            ax.set_title(f'node {inode}')

        # Define legend keyword arguments
        leg_kwargs = dict(
            bbox_to_anchor=(1.0, .5),
            loc='center left',
            frameon=False,
        )

        # If stimulus data exists
        if hasstim:
            # Extract max stimulus intensity per node, and complete column titles
            Isppas = data[self.STIM_KEY].groupby(self.NODE_KEY).max()
            for ax, Isppa in zip(axes[0], Isppas):
                ax.set_title(f'{ax.get_title()} - Isppa = {Isppa:.1f} W/cm2')
            
            # Extract stimulus bounds            
            tvec, _ = self.get_stim_vecs()
            stimbounds = np.array([tvec[1], tvec[-2]])
        
            # If specified, offset time to align 0 with stim onset
            if tref == 'onset':
                data.index = data.index.set_levels(
                    self.extract_time(data) - stimbounds[0], level=self.TIME_KEY)
                stimbounds -= stimbounds[0]
        
            # If specified, mark stimulus span on all axes
            if addstimspan:
                for irow, axrow in enumerate(axes):
                    for ax in axrow:
                        ax.axvspan(
                            *stimbounds, fc='silver', ec=None, alpha=.3,
                            label='bounds' if irow == 0 else None)
        
        # Initialize axis row index
        irow = 0

        # Plot stimulus time-course per node
        if hasstim:
            axrow = axes[irow]
            axrow[0].set_ylabel(self.ISPPA_KEY)
            for ax, (_, stim) in zip(axrow, data[self.STIM_KEY].groupby(self.NODE_KEY)):
                stim.droplevel(self.NODE_KEY).plot(ax=ax, c='k')
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Plot temperature time-course per node
        if hastemps:
            axrow = axes[irow]
            axrow[0].set_ylabel('T (°C)')
            for ax, (_, T) in zip(axrow, data['T'].groupby(self.NODE_KEY)):
                T.droplevel(self.NODE_KEY).plot(ax=ax, c='k')
            irow += 1

        # For each channel type, plot conductance time course per node
        if hasconds:
            axrow = axes[irow]
            if gmode == 'rel':
                ylabel = 'g/g0 (%)'
            elif gmode == 'norm':
                ylabel = 'g/gmax (%)'
            else:
                ylabel = 'g (uS/cm2)'
            axrow[0].set_ylabel(ylabel)
            for condkey in self.record_dict['conductances']:
                if condkey in data:
                    for ax, (_, g) in zip(axrow, data[condkey].groupby(self.NODE_KEY)):
                        g = g.droplevel(self.NODE_KEY)
                        if condkey.endswith('bar'):
                            label = f'\overline{{g_{{{condkey[1:-3]}}}}}'
                        else:
                            label = f'g_{{{condkey[1:]}}}'
                        if gmode == 'rel':
                            if g.iloc[0] == 0:
                                logger.warning(f'Cannot compute relative conductance for {label}: baseline is 0')
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                g = g / g.iloc[0] * 100
                        elif gmode == 'norm':
                            g = g / g.max() * 100
                        g.plot(ax=ax, label=f'${label}$')
            axrow[-1].legend(**leg_kwargs)
            if gmode == 'log':
                for ax in axrow:
                    ax.set_yscale('log')
            irow += 1

        # For each channel type, plot current time course per node
        if hascurrs:
            axrow = axes[irow]
            axrow[0].set_ylabel('I (mA/cm2)')
            for ckey, color in zip(self.record_dict['currents'], plt.get_cmap('Dark2').colors):
                if ckey in data:
                    for ax, (_, i) in zip(axrow, data[ckey].groupby(self.NODE_KEY)):
                        i.droplevel(self.NODE_KEY).plot(ax=ax, label=f'$i_{{{ckey[1:]}}}$', color=color)
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Plot membrane potential time-course per node
        if hasvoltages:
            axrow = axes[irow]
            axrow[0].set_ylabel('Vm (mV)')
            for ax, (_, v) in zip(axrow, data['v'].groupby(self.NODE_KEY)):
                v = v.droplevel(self.NODE_KEY) 
                v.plot(ax=ax, c='k', label='trace')
                aptimes = self.extract_ap_times(v.index, v.values)
                ax.plot(aptimes, np.full_like(aptimes, 70), '|', c='dimgray', label='spikes')
                yb, yt = ax.get_ylim()
                ax.set_ylim(min(yb, -80), max(yt, 50))
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Add figure title, if specified
        if title is None and ncols > 1:
            title = self
        if title is not None:
            fig.suptitle(title)
        
        # Adjust layout
        fig.tight_layout()

        # Return figure
        return fig
    
    def get_stimdist_vector(self, kind='uniform'):
        '''
        Return stimulus distribution vector for a given kind of stimulus distribution.

        :param kind: stimulus distribution kind (default: 'uniform'). One of:
            - "single": single node stimulated
            - "uniform": all nodes stimulated with equal intensity
        '''
        # Check that stimulus distribution kind is valid
        if kind not in ['single', 'uniform']:
            raise ValueError(f'Invalid stimulus distribution kind: {kind}')
        
        # Return stimulus distribution vector
        if kind == 'single':
            x = np.zeros(self.size)
            x[0] = 1
            return x
        elif kind == 'uniform':
            return np.ones(self.size)
        
    def check_stimulus_distribution(self, x):
        '''
        Check that a given stimulus distribution vector is valid.

        :param x: stimulus distribution vector
        '''
        # Check that stimulus distribution vector is valid
        if len(x) != self.size:
            raise ValueError(f'Number of stimulus distribution values {len(x)} does not match number of nodes {self.size}')
        if any(amp < 0 for amp in x):
            raise ValueError('Stimulus distribution values must be positive')
        if sum(x) == 0:
            raise ValueError('At least one stimulus value must be non-zero')
    
    def run_stim_sweep(self, Isppas, stimdist=None, nreps=1, **kwargs):
        ''' 
        Simulate model across a range of stimulus intensities and return outputs.
        
        :param Isppas: range of stimulus intensities (W/cm2)
        :param stimdist (optional): vector spcifying relative stimulus intensities at each node. 
            If not provided, all nodes will be stimulated with the same intensity.
        :param nreps (optional): number of repetitions per stimulus intensity (default: 1)
        :param kwargs: optional arguments passed to "set_stim" and "simulate" methods
        :return: multi-indexed output dataframe with sweep variable, node index and time
        '''
        # If stimulus distribution vector not provided, assume uniform distribution across nodes
        if stimdist is None:
            stimdist = [1] * self.size
        
        # Check that stimulus distribution vector is valid
        self.check_stimulus_distribution(stimdist)
        self.log(f'running simulation sweep across {len(Isppas)} stimulus intensities')

        # Generate 2D array of stimulus vectors for each stimulus intensity
        Isppa_vec_range = np.dot(np.atleast_2d(Isppas).T, np.atleast_2d(stimdist))

        # Disable verbosity during sweep
        vb = self.verbose
        self.verbose = False

        # Initialize empty data list
        data = []

        # Simulate model for each stimulus vector, and append output to data
        tstop = kwargs.pop('tstop', None)
        for Isppa_vec in tqdm(Isppa_vec_range):
            self.set_stim(Isppa_vec, **kwargs)
            data.append(self.simulate(tstop=tstop, nreps=nreps))
        
        # Restore verbosity
        self.verbose = vb

        # Concatenate data with new sweep index level, and return
        return pd.concat(data, keys=Isppas, names=[self.ISPPA_KEY], axis=0)
    
    def plot_sweep_results(self, data, metric, title=None, ax=None, width=4, height=2, legend=True,
                           estimator='mean', errorbar='se'):
        '''
        Plot results of a sweep.
        
        :param data: multi-indexed output dataframe with sweep variable, node index and time
        :param metric: metric(s) to plot
        :param title: optional figure title (default: None)
        :param ax: optional axis handle (default: None)
        :param width: figure width (default: 4)
        :param height: figure height (default: 2)
        :param legend: whether to add a legend to the graph(s)
        :param estimator: estimator for the central tendency (default: 'mean').
        :param errorbar: estimator for the error bars (default: 'se')
        :return: figure handle
        '''
        # If multiple metrics provided, recursively call function for each metric
        if isinstance(metric, (tuple, list)):
            if len(metric) > 1:
                # Create figure
                nmetrics = len(metric)
                fig, axes = plt.subplots(nmetrics, 1, figsize=(width, height * nmetrics), sharex=True)
                if title is not None:
                    fig.suptitle(title)

                # Plot each metric on a separate axis
                for ax, m in zip(axes, metric):
                    self.plot_sweep_results(data, m, ax=ax, legend=legend)
                    legend = False
                
                # Return figure
                return fig
            else:
                metric = metric[0]

        # Create figure and axis, if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Compute metric across nodes and all other dimensions
        mdata = self.compute_metric(data, metric)

        # Extract index dimensions other than "node" and "repetition"
        extradims = [k for k in mdata.index.names if k not in (self.NODE_KEY, self.REP_KEY)]
        
        # If no extra dimension, raise error
        if len(extradims) == 0:
            raise ValueError('Cannot plot sweep results with no extra grouping dimension')
        # If 1 extra dimension, extract sweep key and set extra dimension key to None
        elif len(extradims) == 1:
            sweepkey, extrakey = extradims[0], None
        # If 2 extra dimensions, extract sweep key and extra dimension key
        if len(extradims) == 2:
            try:
                idx = extradims.index(self.ISPPA_KEY)
                sweepkey, extrakey = extradims[idx], extradims[1 - idx]
            except ValueError:
                extrakey, sweepkey = extradims
        # If too many extra dimensions, raise error
        elif len(extradims) > 2:
            raise ValueError('Cannot plot sweep results with more than 2 extra grouping dimensions')

        # If relative change, convert to percentage
        if metric == self.RESP_KEY:
            mdata = mdata * 100

        # If metric data is dataframe, extract mean column and store dataframe
        mvar = None
        if isinstance(mdata, pd.DataFrame):
            mdata, mvar = mdata['mean'].rename(metric), mdata
        
        # Generate color palette
        colors = list(plt.get_cmap('tab10').colors)
        nodes = mdata.index.get_level_values(self.NODE_KEY).unique()
        palette = dict(zip(nodes, colors))

        # Plot metric vs sweep variable
        sns.lineplot(
            data=mdata.to_frame(), 
            ax=ax, 
            x=sweepkey,
            y=metric,
            hue=self.NODE_KEY if len(nodes) > 1 else None,
            marker='o',
            palette=palette if len(nodes) > 1 else None,
            estimator=estimator,
            errorbar=errorbar,
            style=extrakey,
            legend=legend
        )

        # If sweep variance is provided
        if mvar is not None:
            # Compute lower and upper bounds from mean and err columns
            mvar['lb'] = mvar['mean'] - mvar['err']
            mvar['ub'] = mvar['mean'] + mvar['err']
            gby = self.NODE_KEY
            if extrakey is not None:
                gby = [gby, extrakey]
            if 'rep' in mvar.index.names:
                if isinstance(gby, list):
                    gby.append(self.REP_KEY)
                else:
                    gby = [gby, self.REP_KEY]
            for glabel, gdata in mvar.groupby(gby):
                if isinstance(glabel, tuple):
                    glabel = glabel[0]
                ax.fill_between(
                    gdata.droplevel(gby).index, gdata['lb'], gdata['ub'], 
                    fc=palette[glabel], ec=None, alpha=0.3)

        # Move legend outside of plot, if any
        if ax.get_legend() is not None:
            sns.move_legend(
                ax,
                bbox_to_anchor=(1.0, .5),
                loc='center left',
                frameon=False
            )

        # Adjust y-label if needed
        ysuffix = None
        if metric == 'ΔT':
            ysuffix = '(°C)'
        elif metric == 'FR':
            ysuffix = '(Hz)'
        elif metric == self.RESP_KEY:
            ysuffix = '(%)'
        if ysuffix is not None:
            ax.set_ylabel(f'{ax.get_ylabel()} {ysuffix}')

        # Add title, if specified
        if title is not None:
            ax.set_title(title)

        # Return figure
        return fig

    def compute_istim(self, Isppa):
        ''' 
        Compute stimulus-driven current amplitude for a given stimulus intensity

        :param Isppa: stimulus intensity (W/cm2)
        :return: stimulus-driven current amplitude (mA/cm2)
        '''
        # Compute stimulus-driven current amplitude over Isppa range
        iStim = -self.iStimbar * sigmoid(Isppa, self.iStimx0, self.iStimdx)

        # If input is an array, format as series
        if isinstance(Isppa, (tuple, list, np.ndarray)):
            iStim = pd.Series(iStim, index=Isppa, name='iStim (mA/cm2)')
            iStim.index.name = self.ISPPA_KEY

        # Return
        return iStim
    
    def compute_Tmax(self, Isppa, dur=None):
        '''
        Compute temperature reached at the end of a stimulus of 
        specific intensity and duration.

        :param Isppa: stimulus intensity (W/cm2)
        :param dur (optional): stimulus duration (ms)
        :return: maximal temperature reached (°C)
        '''
        # If input is an list/tuple, convert to numpy array
        if isinstance(Isppa, (tuple, list)):
            Isppa = np.asarray(Isppa)

        # If duration not provided, use class attribute
        if dur is None:
            if self.dur is None:
                raise ValueError('No stimulus duration defined')
            dur = self.dur

        # Compute steady-state temperature
        ΔTinf = self.alphaT * Isppa
        Tinf = ΔTinf + self.Tref

        # Compute temperature incrase at the end of the stimulus
        c1 = -self.tauT_abs * np.log((ΔTinf))
        Tmax = Tinf - np.exp(-(dur + c1) / self.tauT_abs)

        # If Isppa or duration is an array, format as series
        if isinstance(Isppa, (tuple, list, np.ndarray)):
            Tmax = pd.Series(Tmax, index=Isppa, name='Tmax (°C)')
            Tmax.index.name = self.ISPPA_KEY
        if isinstance(dur, (tuple, list, np.ndarray)):
            Tmax = pd.Series(Tmax, index=dur, name='Tmax (°C)')
            Tmax.index.name = self.DUR_KEY

        # Return
        return Tmax
    
    def compute_iKTmax(self, Isppa, **kwargs):
        '''
        Compute maximal thermally-activated potassium current amplitude
        for a given stimulus intensity

        :param Isppa: stimulus intensity (W/cm2)
        :return: thermally-activated potassium current amplitude (mA/cm2)
        '''
        # Compute maximal temperature reached
        Tmax = self.compute_Tmax(Isppa, **kwargs)
        
        # Compute potassium current for that temperature
        gKTmax = self.gKT * (Tmax - self.Tref)
        iKTmax = gKTmax * (self.vrest - self.EKT)

        # If Series input, rename
        if isinstance(iKTmax, pd.Series):
            iKTmax.name = 'iKTmax (mA/cm2)'

        # Return
        return iKTmax
    
    def compute_EI_currents(self, Isppa, dur=None):
        '''
        Compute excitatory and inhibitory currents for a given stimulus intensity

        :param Isppa: stimulus intensity (W/cm2)
        :return: dataframe of excitatory and inhibitory currents (mA/cm2)
        '''
        # If duration not provided, use class attribute
        if dur is None:
            if self.dur is None:
                raise ValueError('No stimulus duration defined')
            dur = self.dur

        # Compute stimulus-driven and thermally-activated currents amplitude
        # over Isppa range
        return pd.concat([
            self.compute_istim(Isppa),  # stimulus-driven current
            self.compute_iKTmax(Isppa, dur=dur),  # thermally-activated current at end of stimulus
            self.compute_iKTmax(Isppa, dur=dur / 2).rename('iKT1/2 (mA/cm2)'),  # thermally-activated current at half stimulus duration
        ], axis=1)
    
    def plot_EI_imbalance(self, Isppa, ax=None, style=None, add_Pmap=True, legend=True, **kwargs):
        '''
        Plot the imbalance between excitatory and inhibitory currents 
        over a range of stimulus intensities.

        :param Isppa: stimulus intensity (W/cm2)
        :param ax: optional axis handle (default: None)
        :param style: optional plotting style (default: None)
        :param add_Pmap: whether to add x-axis with pressure values mapped to input Isspa values (default: False)
        :return: figure handle
        '''
        # Compute excitatory and inhibitory currents over Isppa range, and 
        # convert to absolute values
        df = self.compute_EI_currents(Isppa, **kwargs).abs()

        # Remove units from currents names
        df.columns = df.columns.str.rstrip('(mA/cm2)').str.rstrip(' ')

        # Create / retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Plot currents
        ax.set_ylabel('|I| (mA/cm2)')
        colors = {
            'iStim': ('C0', '-'),
            'iKTmax': ('C1', '-'),
            'iKT1/2': ('C1', '--'),
        }
        for k, (c, ls) in colors.items():
            df[k].plot(ax=ax, c=c, ls=ls, label=k)
        if legend:
            ax.legend(bbox_to_anchor=(1.0, .5), loc='center left', frameon=False)

        # Add pressure axis, if requested
        if add_Pmap:
            ax2 = ax.twiny()
            ax2.set_xlabel('P (MPa)')
            ax2.set_xlim(ax.get_xlim())
            Pmax = intensity_to_pressure(Isppa.max() * 1e4) * 1e-6  # MPa
            Prange = np.arange(0, Pmax).astype(int)
            ax2.set_xticks(pressure_to_intensity(Prange * 1e6) / 1e4)  # W/cm2
            ax2.set_xticklabels(Prange)

        # Return
        return fig
        
        
