# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-13 13:37:40
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-10-30 12:03:58

# External imports
import sys
import os
import itertools
import warnings
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neuron import h

# Internal modules imports
from logger import logger
from utils import *


# Matplotlib parameters
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if sys.platform != 'linux':
    matplotlib.rcParams['font.family'] = 'arial'


# Initalize handler for multi-order variable time step integration method
cvode = h.CVode()


def get_start_time(interval, noise=0):
    '''
    Get start time of a pre-synaptic drive, based on its interval and noise parameters.

    :param interval: pre-synaptic drive interval (ms)
    :param noise: noise factor (default: 0)
    :return: start time (ms)
    '''
    start_bounds = np.array([1 - noise, 1 + noise]) * 0.5 * interval  #  start interval bounds
    return np.random.uniform(*start_bounds)  # start time (ms)


def get_NetStim(freq=1, start=None, number=1e9, noise=0):
    '''
    Wrapper around NetStim allowing to set parameters in 1-liner.
    
    :param freq: spiking frequency of pre-synaptic drive (Hz)
    :param start (optional): start time (ms)
    :param number (optional): total number of pre-synaptic spikes
    :return: NetStim object
    '''
    # Compute interval from frequency
    interval = 1e3 / freq  # ms

    # If start time not provided, adjust it to noiuse level
    if start is None:
        start = get_start_time(interval, noise=noise)
    
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


class SimplifiedCorticalNetwork:
    ''' Interface class to a simplified network of cortical neurons. '''

    # Conversion factors
    UM_TO_CM = 1e-4
    NA_TO_MA = 1e-6
    S_TO_US = 1e6
    S_TO_MS = 1e3

    # Default model parameters
    Acell = 11.84e3  # Cell membrane area (um2)
    GKT = 0.58e-9  # S/°C
    gKT_default = GKT / (Acell * UM_TO_CM**2)  # S/cm2
    G_RS_RS = 0.002  # Synaptic weight between RS cells, from Plaksin 2016 (uS)
    g_RS_RS = (G_RS_RS / S_TO_US) / (Acell * UM_TO_CM**2)  # Synaptic weight between RS cells (S/cm2)
    
    Cm = 1.0  # membrane capacitance (uF/cm2)
    mechname = 'RS'  # NEURON mechanism name
    vrest = -71.9  # neurons resting potential (mV)
    DEFAULT_DT = 0.025  # default integration time step (ms)

    TIME_KEY = 'time (ms)'
    STIM_KEY = 'stim'
    NODE_KEY = 'node'
    REP_KEY = 'rep'
    RESP_KEY = 'ΔFR/FR'
    P_MPA_KEY = 'P (MPa)'
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
        },
        'currents': {
            'iStim': 'stimulus-driven current (mA/cm2)',
            'iKT': 'thermally-driven Potassium current (uA/cm2)',
            'iSyn': 'synaptic current (mA/cm2)',
            'iM': 'slow non-inactivating potassium current (mA/cm2)',
            'iDrive': 'driving current (mA/cm2)',
            'iNa': 'sodium current (mA/cm2)',
            'iKd': 'potassium current (mA/cm2)',
            'iLeak': 'leak current (mA/cm2)',
            'iNoise': 'noise current (mA/cm2)',
        },
    }

    CURRENT_KEYS = [c[1:] for c in record_dict['currents'].keys()]

    RECTIFIED_CURRENTS = [  # currents that should be rectified
        'iDrive', 
        'iNa', 
        'iStim'
    ]
    CLIPPED_CURRENTS = [   # currents that should be clipped upon plotting
        'iNa', 
        'iKd', 
        'iLeak', 
        'iDrive', 
        'iSyn', 
        'iM', 
        'iNoise'
    ]                
    ABSOLUTE_CURRENTS = [  # currents that should be converted to current density
        'iDrive', 
        'iSyn'
    ]

    TIMEVAR_SUFFIX = '_t'  # suffix to add to time-varying conductance variables

    CURRENTS_CMAP = {  # Colormap of currents
        'iStim': 'C7',
        'iNa': 'C0',
        'iKd': 'C1',
        'iSyn': 'C2',
        'iM': 'C3',
        'iKT': 'C4',
        'iLeak': 'C5',
    }

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
        allkeys = cls.record_keys(list(cls.record_dict.keys())) + ['iNet']

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

    def __init__(self, nnodes, connect=True, synweight=None, conrate=1., noise_amp=0., verbose=True, **kwargs):
        '''
        Initialize a neural network with a given number of nodes.
        
        :param nnodes: number of nodes
        :param connect: whether to connect nodes (default: True)
        :param synweight: synaptic weight (mS/cm2) (default: None)
        :param conrate: connection rate (default: 1)
        :param noise_amp: noise amplitude (default: 0)
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
            conkwargs = {'fraction': conrate}
            if synweight is not None:
                conkwargs['weight'] = synweight
            self.connect_nodes(**conkwargs)

        # Set model parameters, if provided
        self.mech_params = {}
        self.set_mech_param(**kwargs)

        # Set default stimulus and simulation parameters to None
        self.start = None
        self.dur = None
        self.tstop = None
        self.dt = None

        # Set default noise amplitude
        self.noise_amp = noise_amp

        # Log initialization completion
        self.log('initialized')
    
    def copy(self, nnodes=None):
        '''
        Return a copy of the network.
        
        :param nnodes: number of nodes (default: None, i.e. same as original network)
        :return: SimplifiedCorticalNetwork instance
        '''
        # If number of nodes not provided, use same as original network
        if nnodes is None:
            nnodes = self.size

        # Initialize dictionary of parameters to pass to new instance
        initkwargs = {
            'verbose': self.verbose,
            'connect': False,
        }

        # If connected, assign same synaptic weight and connection rate as original network
        if self.is_connected():
            initkwargs['synweight'] = self.get_synaptic_weight()
            initkwargs['conrate'] = self.get_connection_rate()
            initkwargs['connect'] = True
        
        # Create new instance of the class
        model = self.__class__(
            nnodes, 
            **initkwargs,
            **self.mech_params
        )

        # If drive is set, assign same pre-synaptic drive parameters as original network
        if self.is_drive_set():
            model.set_presyn_drive(**self.get_drive_params())
        
        # Assign same stimulus parameters as original network
        model.start = self.start
        model.dur = self.dur
        model.tstop = self.tstop

        # Assign same simulation parameters as original network
        model.tstop = self.tstop
        model.dt = self.dt

        # Assign same noise amplitude as original network
        model.noise_amp = self.noise_amp

        # Return new instance
        return model
        
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
        self.connections = None
    
    @property
    def size(self):
        return len(self.nodes)

    @property
    def inodes(self):
        ''' Return array of node indexes. '''
        return np.arange(self.size)

    @property
    def nodelist(self):
        ''' Return list of node names. '''
        return [node.name() for node in self.nodes]

    @property
    def refarea(self):
        ''' Surface area of first node in the model (um2). '''
        return self.nodes[0](0.5).area()  # um2
    
    def set_biophysics(self):
        ''' Assign membrane capcitance and membrane mechanisms to all nodes. '''
        for node in self.nodes:
            node.cm = self.Cm
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
    
    def parse_node_index(self, inode):
        '''
        Parse a node index.

        :param inode: node index (int, list, tuple or None)
        :return: parsed list of node indexes
        '''
        # If no node index provided, return array of all node indexes
        if inode is None:
            return self.inodes
        # Otherwise, return provided node index(es) as array
        else:
            return np.asarray(as_iterable(inode))
    
    def _set_mech_param(self, key, value, inode=None):
        '''
        Set a specific mechanism parameter on a specific set of nodes.
        
        :param key: parameter name
        :param value: parameter value
        :param inode: node index (default: None, i.e. all nodes)
        '''
        # Get default parameter value
        default_val = self.get_mech_param(key)

        # Assign value to dictionary of mechanism parameters
        self.mech_params[key] = value

        # If value no different than default, return
        if value == default_val:
            return

        # Parse node index
        inode = self.parse_node_index(inode)
        
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
        Convert conductance density to absolute conductance.

        :param g: conductance density, expressed in NEURON units (S/cm2)
        :return: absolute conductance, expressed in NEURON units (uS)
        '''
        # Get cell membrane area in cm2
        A = self.Acell * self.UM_TO_CM**2  # cm2
        # Multiply by neuron soma membrane area
        G = g * A  # S
        # Convert to uS, and return
        return G * self.S_TO_US  # uS
    
    def to_conductance_density(self, G):
        '''
        Convert absolute conductance to conductance density.

        :param G: absolute conductance, expressed in NEURON units (uS)
        :return: conductance density, expressed in NEURON units (S/cm2)
        '''
        # Convert conductance to S
        G = G / self.S_TO_US  # S
        # Get cell membrane area in cm2
        A = self.Acell * self.UM_TO_CM**2  # cm2
        # Divide by neuron soma membrane area, and return
        return G / A  # S/cm2

    def to_current_density(self, I):
        '''
        Convert current to current density.

        :param I: absolute current, expressed in NEURON units (nA)
        :return: current density, expressed in NEURON units (mA/cm2)
        '''
        # Convert current to mA
        I = I * self.NA_TO_MA  # mA
        # Get cell membrane area in cm2
        A = self.Acell * self.UM_TO_CM**2  # cm2
        # Divide by neuron soma membrane area, and return
        return I / A  # mA/cm2
    
    def to_current(self, i):
        '''
        Convert current density to current for compatibility with NEURON formalism.

        :param i: current density, expressed in NEURON units (mA/cm2)
        :return: absolute current, expressed in NEURON units (nA)
        '''
        # Get cell membrane area in cm2
        A = self.Acell * self.UM_TO_CM**2  # cm2
        # Multiply by neuron soma membrane area
        I = i * A  # mA
        # Convert to nA, and return
        return I / self.NA_TO_MA  # nA
    
    def set_presyn_drive(self, inode=None, delay=0, weight=4.5e-4, sync=False, **kwargs):
        ''' 
        Set a presynaptic drive on a specific set of nodes.

        :param inode: node index (default: None, i.e. all nodes)
        :param delay: synaptic delay between pre-synaptic drive and target nodes (ms)
        :param weight: synaptic weight, i.e. maximal synaptic conductance (S/cm2)
        :param sync: whether to synchronize pre-synaptic drive across nodes (default: False)
        :param kwargs: dictionary of parameter names and values fed to "get_NetStim" function
        '''
        # Parse node index(es)
        inode = self.parse_node_index(inode)
        
        self.log(f'setting pre-synaptic drive on nodes {inode}')

        # If sync flag enabled, generate unique pre-synaptic drive with passed parameters
        if sync:
            stim = get_NetStim(**kwargs)
        else:
            stim = None

        # For each target node
        for i in inode:
            if self.is_drive_set(inode=i):
                self.log(f'modifying pre-synaptic drive on node {i}', warn=True)
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
        # Parse node index(es)
        inode = self.parse_node_index(inode)

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
        params.drop('start', axis=1, inplace=True)
        if any(params.nunique(axis=0) != 1):
            raise ValueError('pre-synaptic drive parameters differ across nodes')
        return params.iloc[0].to_dict()

    def is_drive_set(self, inode=None, as_scalar=True):
        ''' 
        Check if presynaptic drive is set on a specific set of nodes

        :param inode: node index (default: None, i.e. all nodes)
        :param as_scalar: whether to return a scalar (default: True)
        :return: boolean(s) indicating whether drive is set on specified node(s) 
        '''
        # Parse node index(es)
        inode = self.parse_node_index(inode)
        
        # Check status of pre-synaptic drive in all nodes
        out = [self.drive_objs[i] is not None for i in inode]

        # If requested, return scalar
        if as_scalar:
            out = all(out)

        # Return
        return out
    
    def update_drive_starts(self):
        ''' Update start time of all drives '''
        # If no pre-synaptic drive set, raise error
        if not self.is_drive_set():
            raise ValueError('no pre-synaptic drive set')
        # Loop through all drives
        for i, drive in enumerate(self.drive_objs):
            # Update NetStim start time
            drive[0].start = get_start_time(drive[0].interval, noise=drive[0].noise)
            self.log(f'setting start time of pre-synaptic drive {i} to {drive[0].start:.2f} ms...')

    def remove_presyn_drive(self, inode=None):
        ''' 
        Remove presynaptic drive on a specific set of nodes

        :param inode: node index (default: None, i.e. all nodes)
        '''
        # Parse node index(es)
        inode = self.parse_node_index(inode)

        # For each target node
        for i in inode:
            if not self.is_drive_set(inode=i):
                self.log(f'no pre-synaptic drive to remove at node {i}', warn=True)
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

        # Append synapse and netcon objects to connections atribute 
        self.connections[ipresyn, ipostsyn] = (syn, nc)    
    
    def connect_nodes(self, fraction=1., **kwargs):
        ''' 
        Form specific connections between network nodes
        
        :param fraction: fraction of nodes to connect (default: 1)
        '''
        # Check that fraction is between 0 and 1
        if not is_within(fraction, (0, 1)):
            raise ValueError(f'invalid connection fraction: {fraction}')
        
        # List all candidate node pairs (exluding self-connections)
        candidate_pairs = np.array(list(itertools.permutations(self.inodes, 2)))
        ncandidates = candidate_pairs.shape[0]

        # Determine number of connections to be made according to the specified fraction
        nconns = int(np.round(fraction * ncandidates))

        # Select a random subset of node pairs to connect
        idxs = np.sort(np.random.choice(np.arange(ncandidates), nconns, replace=False))
        pairs = candidate_pairs[idxs]

        # Compute effective fraction of selected node pairs, and check that it is within 1% of the requested fraction
        frac = nconns / ncandidates
        if np.abs(frac - fraction) > 0.01:
            raise ValueError(f'computed connection fraction ({fraction}) diverges from requested fraction ({frac})')
        self.log(f'connecting {frac * 1e2:.1f}% ({nconns}/{ncandidates}) of candidate node pairs')

        # Initialize connections 2D array
        self.connections = np.full((self.size, self.size), None)

        # For each selected node pair
        for pair in pairs:
            self.connect(*pair, **kwargs)
    
    def is_connected(self):
        ''' Return whether network nodes are connected. '''
        return hasattr(self, 'connections') and self.connections is not None
    
    def disconnect_nodes(self):
        ''' Clear all synapses and network connection objects between nodes '''
        self.log('removing all connections between nodes')
        self.connections = None
    
    def set_synaptic_weight(self, w):
        ''' 
        Set synaptic weight on all connections.
        
        :param w: synaptic weight (S/cm2)
        '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        self.log(f'setting all synaptic weights to {w:.2e} S/cm2')
        for con in np.ravel(self.connections):
            if con is not None:
                con[1].weight[0] = self.to_absolute_conductance(w)  # synaptic weight (uS)
    
    def get_synaptic_weight(self):
        '''
        Return synaptic weight on all connections.
        
        :return: synaptic weight (S/cm2)
        '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        for con in np.ravel(self.connections):
            if con is not None:
                return self.to_conductance_density(con[1].weight[0])
        raise ValueError('No synaptic weight found')
    
    def set_connection_rate(self, fraction):
        ''' 
        Set connection rate between nodes.
        
        :param fraction: fraction of nodes to connect (0-1)
        '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        self.disconnect_nodes()
        self.connect_nodes(fraction=fraction)

    def get_connection_rate(self):
        '''
        Return connection rate between nodes.
        
        :return: connection rate (0-1)
        '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        return np.count_nonzero(self.connections) / self.connections.size
    
    @staticmethod
    def get_probe(var):
        '''
        Return a NEURON vector probe for a specific variable.
        '''
        probe = h.Vector()
        probe.record(var)
        return probe    
    
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

        # If variable is synaptic current
        if varname == 'iSyn':
            if self.connections is not None:
                logger.debug(f'recording post-synaptic currents')
                # Initialize recording probes list for synaptic currents
                self.probes[varkey] = []
                # For each post-synaptic node
                for ipostnode in self.inodes:
                    plist = []
                    # For each pre-synaptic node
                    for iprenode in self.inodes:
                        # If connection exists
                        if self.connections[iprenode, ipostnode] is not None:
                            # record synaptic current
                            plist.append(
                                self.get_probe(self.connections[iprenode, ipostnode][0]._ref_i))
                    self.probes[varkey].append(plist)
        
        # If variable is noise current 
        elif varname == 'iNoise':
            pass

        # Otherwise
        else:
            # Record variable on all nodes
            logger.debug(f'recording "{varname}" on all nodes with key "{varkey}"')
            self.probes[varkey] = [
                self.get_probe(self.get_var_ref(inode, varname)) for inode in self.inodes]
    
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
        logger.debug(f'disabled currents: {", ".join([ikey for _, ikey in l])}')
        is_drive_set = self.is_drive_set()
        if isinstance(is_drive_set, list):
            is_drive_set = all(is_drive_set)
        if not is_drive_set:
            l.append(('iDrive',))
        if not self.is_stim_set():
            l.append(('iStim',))
        if self.noise_amp == 0.:
            l.append(('iNoise',))
        return l
    
    def get_disabled_keys(self):
        ''' List all keys associated to disabled currents '''
        return list(itertools.chain(*self.get_disabled_currents()))
    
    def record_continous_variables(self):
        ''' Set probes to record time-varying continuous variables on all nodes. '''
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
    
    def record_presyn_drive(self):
        ''' Record event times of all pre-synaptic drives. '''
        if not self.is_drive_set():
            raise ValueError('no pre-synaptic drive set')
        self.drives_probes = {}
        for node, drive in zip(self.nodelist, self.drive_objs):
            self.drives_probes[node] = h.Vector()
            drive[-1].record(self.drives_probes[node])
    
    def extract_presyn_events(self):
        '''
        Extract event times of all pre-synaptic drives.
        
        :return: pandas series of event times per node (in ms)
        '''
        if not self.is_drive_set():
            raise ValueError('no pre-synaptic drive set')
        # Compute dictionary of event times per node 
        events = {k: np.array(v.to_python()) for k, v in self.drives_probes.items()}
        # Return as pandas series
        s = pd.Series(events)
        s.index.name = self.NODE_KEY
        s.name = 'event times (ms)'
        return s
    
    def extract_continous_variables(self):
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
            if k == 'iSyn':
                # Initialize 2D array of synaptic current time course across nodes
                vpernode = np.zeros((self.size, len(t)))
                # For each post-synaptic node
                for ipost, probes in enumerate(v):
                    # Extract 2D array of synaptic current time course across pre-synaptic nodes
                    vpres = np.array([x.to_python() for x in probes])
                    # Sum across pre-synaptic nodes to obtain total synaptic current flowing into post-synaptic node
                    vpernode[ipost] = np.sum(vpres, axis=0)
            else:
                # Extract 2D array of variable time course across nodes
                vpernode = np.array([x.to_python() for x in v])

            # If probe recorded absolute current, convert to current density
            if k in self.ABSOLUTE_CURRENTS:
                vpernode = self.to_current_density(vpernode)
            
            # Create multi-indexed series from 2D array, and append to output dataframe
            dfout[k] = pd.DataFrame(
                vpernode.T, 
                columns=self.nodelist, 
                index=t
            ).T.stack()

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
        # If multi-indexed dataframe
        if isinstance(data.index, pd.MultiIndex):
            # Extract time vector of first node (or first group of other dimensions)
            mux_slice = list(data.index[0])
            itime = data.index.names.index(cls.TIME_KEY)
            mux_slice[itime] = slice(None)
            return data.loc[tuple(mux_slice)].index.values  # ms
        # If single-indexed dataframe
        else:
            # Check that index is time
            if data.index.name != cls.TIME_KEY:
                raise ValueError(f'cannot extract time: data index = {data.index.name}')
            # Return time vector
            return data.index.values
    
    @classmethod
    def extract_var(cls, data, key):
        '''
        Extract variable time course across nodes from simulation output dataframe
        
        :param data: multi-indexed simulation output dataframe
        :param key: variable name
        :return: 2D array of variable time course across nodes
        '''
        return np.array([s.values for _, s in data[key].groupby(cls.NODE_KEY)])
        # return data[key].unstack().values

    def get_stim_waveform(self, start, dur, amp, stepdt=0.):
        ''' 
        Define a stimulus waveform as a vector of time/amplitude pairs, 
        based on global stimulus parameters.

        :param start: stimulus start time (ms)
        :param dur: stimulus duration (ms)
        :param amp: stimulus amplitude
        :param stepdt: time step for stimulus step transitions (ms)
        :return: (time - amplitude) waveform vector
        '''
        return np.array([
            [0, 0],  # start at 0
            [start - stepdt / 2, 0],  # hold at 0 until stimulus onset
            [start + stepdt / 2, amp],  # switch to amplitude
            [start + dur - stepdt / 2, amp],  # hold at amplitude until stimulus offset
            [start + dur + stepdt / 2, 0],  # switch back to 0
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
            # Identify common suffix of all columns
            suffix = os.path.commonprefix([c[::-1] for c in values.columns])[::-1]
            err = values[f'err{suffix}'].values
            values = values[f'mean{suffix}'].values

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

    def set_stim(self, A, start=None, dur=None, fraction=1.):
        ''' 
        Set stimulus per node node with specific waveform parameters.
        
        :param A: stimulus amplitude scalar, or vector of amplitudes per node
        :param start: stimulus start time (ms)
        :param dur: stimulus duration (ms)
        :param fraction: fraction of nodes to stimulate (default: 1)
        '''
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

        # If scalar amplitude provided, convert fo amplitudes vector using specified fraction
        if isinstance(A, (int, float)):
            amps = self.get_stimdist_vector(kind=fraction) * A

        # Otherwise, set amplitudes vector
        else:
            amps = np.asarray(A)

        # Check that stimulus amplitudes are valid
        if len(amps) != self.size:
            raise ValueError(f'Number of stimulus amplitudes ({len(amps)}) does not match number of nodes {self.size}')
        if any(amp < 0 for amp in amps):
            raise ValueError('Stimulus amplitude must be positive')
        
        # Compute fraction of nodes that will be stimulated
        frac = np.mean(amps > 0)
        A = amps.max()
        self.log(f'setting {self.dur:.2f} ms stimulus with amplitude {A:.2f} MPa on {frac * 1e2:.1f}% of nodes')
        # amps_str = self.vecstr(amps, suffix='MPa')
        # self.log(f'setting {self.dur:.2f} ms stimulus with node-specific amplitudes:\n{amps_str}')
        
        # Set stimulus vectors
        self.h_yvecs = []
        for amp in amps:
            tvec, yvec = self.get_stim_waveform(self.start, self.dur, amp).T
            self.h_yvecs.append(h.Vector(yvec))
        self.h_tvec = h.Vector(tvec)

        # Play stimulus on all nodes with node-specific amplitudes
        for inode in self.inodes:
            self.h_yvecs[inode].play(
                self.get_var_ref(inode, 'Pamp'), self.h_tvec, True)
    
    def is_stim_set(self):
        ''' Return whether stimulus is set. '''
        return hasattr(self, 'h_tvec') and hasattr(self, 'h_yvecs')
    
    def get_stim_vecs(self):
        ''' Return stimulus time-course and amplitude vectors. '''
        tvec = np.array(self.h_tvec.to_python())
        stimvecs = np.array([y.to_python() for y in self.h_yvecs])
        return tvec, stimvecs
    
    def interpolate_vector(self, t, y, teval):
        ''' 
        Interpolate waveform along a set of evaluation times
        
        :param tref: time vector (ms)
        :param y: waveform vector
        :param teval: evaluation time vector (ms)
        '''
        return interp1d(
            t, y, 
            kind='previous', 
            bounds_error=False,
            fill_value=(0., y[-1])
        )(teval)
    
    def interpolate_vectors(self, tvec, yvecs, teval):
        ''' 
        Interpolate waveforms along a set of evaluation times
        
        :param tvec: time vector (ms)
        :param yvecs: waveform vectors
        :param teval: evaluation time vector (ms)
        :return: multi-indexed series storing the interpolated waveform time courses across nodes
        '''
        # Create output dataframe, indexed by time
        df = pd.DataFrame(index=teval)
        # For each node
        for node, yvec in zip(self.nodelist, yvecs):
            # Add column with interpolated waveform along evaluation time vector
            df[node] = self.interpolate_vector(tvec, yvec, teval)
        # Stack dataframe and return as series
        return df.T.stack()
    
    def interpolate_stim_data(self, teval):
        '''
        Interpolate stimulus waveform along simulation time vector

        :param teval: evaluation time vector (ms)
        :return: multi-indexed series storing the interpolated time course
            of the stimulus across nodes
        '''
        out = self.interpolate_vectors(
            *self.get_stim_vecs(), 
            teval
        ).rename(self.STIM_KEY)
        return out
    
    def interpolate_noise_data(self, teval):
        '''
        Interpolate noise current waveform along simulation time vector

        :param teval: evaluation time vector (ms)
        :return: multi-indexed series storing the interpolated time course
            of the noise current across nodes
        '''
        return self.interpolate_vectors(
            np.array(self.noise_tvec.to_python()), 
            self.to_current_density(np.array([x.to_python() for x in self.noise_ivecs])),  # mA/cm2
            teval
        ).rename('iNoise')
    
    def remove_stim(self):
        ''' Remove stimulus from all nodes. '''
        if not self.is_stim_set():
            self.log('no stimulus to remove', warn=True)
            return
        self.log('removing stimulus')
        for inode in self.inodes:
            self.h_yvecs[inode].play_remove()
        del self.h_tvec
        del self.h_yvecs
        self.start = None
        self.dur = None
    
    def concatenate_outputs(self, inputs, outputs, key):
        '''
        Concatenate simulation output dataframes across repetitions.

        :param inputs: list of concatenation values
        :param outputs: list of simulation outputs
        :param key: name of concatenation dimension
        '''
        # Check that number of outputs matches number of concatenation inputs
        if len(outputs) != len(inputs):
            raise ValueError(f'number of outputs ({len(outputs)}) does not match number of values ({len(inputs)})')
        
        # Separate timeseries from events data, if present 
        if isinstance(outputs[0], tuple):
            timeseries, evts = list(zip(*outputs))
        else:
            timeseries, evts = outputs, None
        
        # Cast concatenation key as list if not already
        key = as_iterable(key)
        
        # Concatenate timeseries dataframes
        data = pd.concat(timeseries, axis=0, keys=inputs, names=key)
        
        # Concatenate events dataframes, if present
        if evts is not None:
            evts = pd.concat(evts, axis=0, keys=inputs, names=key)
            data = (data, evts)
        
        # Return concatenated data
        return data
    
    @property
    def noise_amp(self):
        ''' Get noise current amplitude (mA/cm2) '''
        return self._noise_amp
    
    @noise_amp.setter
    def noise_amp(self, val):
        '''
        Set noise current amplitude

        :param A: noise current amplitude (mA/cm2)
        '''
        if val < 0:
            raise ValueError('noise amplitude must be positive or null')
        self._noise_amp = val  # mA/cm2
    
    def get_noise_vector(self, tvec):
        '''
        Generate Gaussian noise current vector.

        :param tvec: time vector (ms)
        :return: noise current vector (mA/cm2)
        '''
        return np.random.normal(0., self.noise_amp, len(tvec))
    
    def inject_noise_to_node(self, inode, tvec):
        '''
        Inject noise current to a specific node.

        :param inode: node index
        :param tvec: time vector (ms)
        :return: noise current vector and current current clamp object
        '''
        # Generate Gaussian noise vector, converted to nA units 
        noise_vec = h.Vector(self.to_current(self.get_noise_vector(tvec)))

        # Create current clamp object 
        stim = h.IClamp(self.nodes[inode](0.5))
        stim.delay = 0.0
        stim.dur = 1e9
        
        # Play noise current into current clamp
        noise_vec.play(stim._ref_amp, tvec, True)

        # Return noise current vector and current clamp object
        return noise_vec, stim
 
    def inject_noise(self, dt):
        '''
        Inject noise current all nodes.

        :param dt: simulation time step (ms)
        '''
        # Generate noise time vector
        self.noise_tvec = h.Vector(np.arange(0, self.tstop + dt / 2, dt))

        # Initialize noise current vectors and current clamp objects
        self.noise_ivecs = []
        self.noise_iclamps = []

        # Loop through model nodes
        for inode in self.inodes:
            # Inject noise current to node
            noise_vec, stim = self.inject_noise_to_node(inode, self.noise_tvec)

            # Append noise current vector and current clamp object to network class atributes
            self.noise_ivecs.append(noise_vec)
            self.noise_iclamps.append(stim)
    
    def has_noise(self):
        ''' Return whether noise current is set. '''
        return hasattr(self, 'noise_tvec')
    
    def remove_noise(self):
        '''
        Remove noise current from all nodes.
        '''
        if not self.has_noise():
            self.log('no noise current to remove')
            return
        for inode in self.inodes:
            self.noise_ivecs[inode].play_remove()
            self.noise_iclamps[inode].amp = 0.
        del self.noise_iclamps
        del self.noise_ivecs
        del self.noise_tvec
        
    def simulate(self, tstop=None, nreps=1, dt=None):
        '''
        Run a simulation for a given duration.
        
        :param tstop: simulation duration (ms)
        :param nreps: number of simulation repetitions (default: 1)
        :param dt: simulation time step (ms)
        :return: (time vector, voltage-per-node array) tuple
        '''
        # If multiple repetitions requested, recursively call function
        if nreps > 1:
            ireps = range(nreps)
            data = []
            for irep in ireps:
                self.log(f'repetition {irep + 1}/{nreps}')
                data.append(self.simulate(tstop=tstop, dt=dt))
            return self.concatenate_outputs(ireps, data, self.REP_KEY)
        
        # If time step provided, update class attribute
        if dt is not None:
            self.dt = dt
        
        # If simulation duration provided, update class attribute
        if tstop is not None:
            self.tstop = tstop
        
        # If no simulation duration defined, raise error
        if self.tstop is None:
            raise ValueError('No simulation duration defined')
        
        # Check that simulation duration outlasts stimulus waveform
        if self.is_stim_set() and self.tstop < self.h_tvec.x[-1]:
            raise ValueError('Simulation duration must be longer than stimulus waveform offset')
        
        # If pre-synaptic drive is set
        if self.is_drive_set():
            # Update start time of all drives
            self.update_drive_starts()
            # Record pre-synaptic drive events
            self.record_presyn_drive()
        
        # Initialize recording probes
        self.record_continous_variables()

        # If noise amplitude > 0
        if self.noise_amp > 0:
            # If dt is not provided, raise warning and set dt to default value
            if self.dt is None:
                self.log(
                    f'noise current injection requires fixed time step -> setting dt = {self.DEFAULT_DT} ms', warn=True)
            self.dt = self.DEFAULT_DT
            # Inject noise current to all nodes
            self.inject_noise(self.dt)
        else:
            if dt is None and self.dt is not None:
                self.log('no noise current injection -> switching to variable dt', warn=True)
                self.dt = None

        # Set simulation time step (if provided), or set up variable time step integration
        if self.dt is not None:
            h.dt = self.dt  # ms
            cvode.active(0)
        else:
            cvode.active(1)
            cvode.maxstep(5)

        # Run simulation
        self.log(f'simulating for {self.tstop:.2f} ms')
        h.finitialize(self.vrest)
        while h.t < self.tstop:
            h.fadvance()
        
        # Extract dataframe of recorded continous variables 
        # self.log('extracting output results')
        data = self.extract_continous_variables()
        
        # If stimulus is set, interpolate stimulus waveforms along simulation time vector
        if self.is_stim_set():
            data[self.STIM_KEY] = self.interpolate_stim_data(self.extract_time(data))

        # If noise is set
        if self.has_noise():
            # Interpolate noise current waveforms along simulation time vector
            data['iNoise'] = self.interpolate_noise_data(self.extract_time(data))
            # Remove noise current from all nodes
            self.remove_noise()

        # Compute and add net current (without contribution of noise current)
        ckeys = [k for k in self.record_dict['currents'] if k in data and k!= 'iNoise']
        data['iNet'] = data[ckeys].sum(axis=1).rename('iNet')

        # If drive is set, extract dictionary of presynaptic drive events, and add it to data
        if self.is_drive_set():
            events = self.extract_presyn_events()
            data = (data, events)

        # # Compute and log max temperature increase per node
        # ΔT = self.compute_metric(data, 'ΔT')
        # self.log(f'max temperature increase:\n{self.vecstr(ΔT, prefix="ΔT =", suffix="°C")}')
        
        # # Count number of spikes and average firing rate per node
        # nspikes = self.compute_metric(data, 'nspikes')
        # self.log(f'number of spikes:\n{self.vecstr(nspikes, prefix="n =", suffix="spikes")}')
        # FRs = self.compute_metric(data, 'FR')
        # self.log(f'firing rate:\n{self.vecstr(FRs, prefix="FR =", suffix="Hz")}')

        # # Compute number of non-artificial spikes per node
        # if self.is_drive_set():
        #     nspikes_evoked = self.compute_metric(data, 'nspikes_evoked')
        #     self.log(f'number of non-artificial spikes:\n{self.vecstr(nspikes_evoked, prefix="n =", suffix="spikes")}')

        # # If stimulus is set, compute stimulus-evoked response
        # if self.is_stim_set():
        #     stimresp = self.compute_metric(data, self.RESP_KEY)
        #     self.log(f'stimulus response:\n{self.vecstr(stimresp * 100, prefix=f"{self.RESP_KEY} =", suffix="%")}')
        
        # Return continous variables and potential events
        return data
    
    def is_excited(self, Pamp):
        '''
        Check whether network is excited (i.e., at least 1 AP in at least 1 node) at a given stimulation amplitude

        :param Pamp: peak pressure amplitude (MPa)
        :return: boolean
        '''
        self.set_stim(Pamp)
        data = self.simulate()
        return self.compute_metric(data, 'nspikes').max() > 0
    
    def find_spiking_threshold(self, Pmax=5., rtol=.05):
        '''
        Find threshold amplitude driving a network response (i.e., at least 1 AP in at least 1 node)

        :param Pmax: upper bound of pressure amplitude search range (MPa)
        :param rtol: relative tolerance for threshold estimation, i.e. (Pout - Pthr) / Pthr < rtol
        :return: threshold pressure amplitude (MPa)
        '''
        # Define search interval
        Pbounds = [0, Pmax]

        self.log(f'finding response thresold pressure amplitude within {Pbounds} MPa interval')
        # Turn off verbosity temporarily
        vb = self.verbose
        self.verbose = False

        # Check that upper bound pressure amplitude generates at least 1 spike 
        if not self.is_excited(Pmax):
            raise ValueError(f'no response detected up to P = {Pmax:.2f} MPa')

        # Set up binary search
        conv = False
        while not conv:
            rel_var = (Pbounds[1] - Pbounds[0]) / Pbounds[1]
            I = (Pbounds[0] + Pbounds[1]) / 2
            if self.is_excited(I):
                if rel_var < rtol:
                    conv = True
                Pbounds[1] = I
            else:
                Pbounds[0] = I
        
        # Reset verbosity
        self.verbose = vb

        # Return threshold amplitude
        return I
    
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
    
    def detect_artificial_spikes(self, tspikes, events, max_delay=10):
        '''
        Detect "artificial" spikes (i.e. spikes evoked by pre-synaptic drive) based on
        relative timing of spikes and pre-synaptic drive events.

        :param tspikes: list/array of spike times (ms)
        :param events: list/array of pre-synaptic drive events (ms)
        :param max_delay: maximal allowed delay between pre-synaptic event and spike time for
         the spike to be classified as "artificial" (ms)
        :return: boolean array indicating whether each spike was evoked by 
            the pre-synaptic drive 
        '''
        # If events input is a dictionary
        if isinstance(events, (dict, pd.Series)):
            # Check that tspikes is also 2-dimensional
            if not isinstance(tspikes[0], (list, np.ndarray)):
                raise ValueError('events dictionary provided but tspikes is 1-dimensional')
            # Initialize output dictionary
            is_artificial = {}
            # Call function recursively for each node
            for ts, (node, evts) in zip(tspikes, events.items()):
                is_artificial[node] = self.detect_artificial_spikes(ts, evts, max_delay=max_delay)
            if isinstance(events, pd.Series):
                is_artificial = pd.Series(is_artificial, name='is_artificial')
            # Return output dictionary
            return is_artificial
        
        # Construct 2D meshgrid of events and spikes 
        TS, EVTS = np.meshgrid(tspikes, events, indexing='ij')
    
        # Compute 2D array of delays between each spike and each pre-synaptic event
        tdiff = TS - EVTS
    
        # Identify spikes whose delay is within specified range
        is_close = np.logical_and(tdiff < max_delay, tdiff > 0)
        ispikes, ievts = np.where(is_close)
        df = pd.DataFrame(data={'ispike': ispikes, 'ievt': ievts})
    
        # Identify and remove duplicates in "ievt" column, in case 2 successive spikes
        # were linked to the same pre-synaptic event
        df = df.drop_duplicates(subset='ievt')
    
        # Extract spikes indexes
        ispikes = df['ispike'].values
        
        # Create boolean array indicating whether each spike was evoked by 
        # the pre-synaptic drive
        is_artificial = np.zeros_like(tspikes, dtype=bool)
        is_artificial[ispikes] = True

        # Return output boolean dictionary
        return is_artificial  
    
    def extract_stim_response(self, data):
        '''
        Extract stimulus-evoked response from simulation output, as the relative 
        change in firing rate between the response window and the rest of the 
        simulation duration.

        :param data: multi-indexed simulation output dataframe
        :return: array of stimulus response per node 
        '''
        # Check that stimulus is set
        if not self.is_stim_set():
            raise ValueError('cannot extract evoked response: stimulus not set')
        
        # Define response analysis windows
        windows = {
            'pre': (0, self.start),  # pre: from simulation start to stim onset
            'peri': (self.start, self.start + self.dur + 10.),  # peri: from stim onset to 10 ms post-stim
            'post': (self.start + self.dur + 10., self.tstop)  # post: from 10 ms post-stim to simulation end
        }

        # Extract list of spikes timings per node
        tspikes_per_node = self.extract_ap_times(
            self.extract_time(data), self.extract_var(data, 'v'))

        # For each node
        stimresps = []
        for tspikes in tspikes_per_node:
            # Convert spikes list to numpy array
            tspikes = np.asarray(tspikes)
            # Classify spikes by analysis window
            tspikes_dict = {k: tspikes[is_within(tspikes, w)] for k, w in windows.items()}
            # Compute firing rates in each window
            FRs = {}
            for k, ts in tspikes_dict.items():
                if ts.size > 1:
                    FRs[k] = np.mean(1 / np.diff(ts)) * self.S_TO_MS  # if more than 1 spike, compute average of ISI reciprocal (Hz)
                else:
                    FRs[k] = np.nan  # otherwise, set firing rate to NaN
            # Compute baseline firing rate as average of pre-stim and post-stim firing rates
            FR0 = np.nanmean([FRs['pre'], FRs['post']])
            # Extract peri-stimulus firing rate
            FR = FRs['peri']
            # If less than 2 spikes during peri-stim window, compute FR using last pre-stim spike
            #  and first peri-stim spike
            if np.isnan(FR):
                if len(tspikes_dict['pre']) == 0 or len(tspikes_dict['peri']) == 0:
                    FR = np.nan
                else:
                    tpre = tspikes_dict['pre'][-1]
                    tperi = tspikes_dict['peri'][0]
                    FR = 1 / (tperi - tpre) * self.S_TO_MS  # Hz
            # If no FR for either baseline or stim, set stimulus response to None
            if np.isnan(FR0) or np.isnan(FR):
                stimresp = None
            # Otherwise, compute stimulus response as stim-evoked relative change in firing rate
            else:
                stimresp = (FR - FR0) / FR0

            # # Classify spikes as during our outside stimulus window
            # is_stim_on = is_within(tspikes, stimbounds)
            # # Compute firing rate during stimulus window
            # FR_stim = is_stim_on.sum() / self.dur * 1000  # Hz
            # # Compute firing rate outside stimulus window
            # FR_base = (tspikes.size - is_stim_on.sum()) / (self.tstop - self.dur) * 1000  # Hz
            # # If no spikes detected outside window, set stimulus response to None
            # if FR_base == 0:
            #     stimresp = None
            # # Otherwise, compute stimulus response as stim-evoked relative change in firing rate
            # else:
            #     stimresp = (FR_stim - FR_base) / FR_base
            
            # Append stimulus response to list
            stimresps.append(stimresp)
        
        # Return list of stimulus response per node
        return stimresps
    
    @staticmethod
    def nspikes_to_peak_dFF(n):
        ''' Convert number of spikes to peak dFF. '''
        return sigmoid(n, 15, 3)
    
    @classmethod
    def plot_peak_dFF_vs_nspikes(cls, nmax=50, ax=None):
        ''' 
        Plot peak dFF vs number of spikes.
        
        :param nmax: maximal number of spikes (optional)
        :param ax: axis to plot on (optional)
        :return: figure handle
        '''
        # Define number of spikes vector
        nspikes = np.linspace(0, nmax, 100)

        # Convert to peak dFF vector
        dFF = cls.nspikes_to_peak_dFF(nspikes)

        # Construct/retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)
                
        # Plot peak dFF vs number of spikes
        ax.plot(nspikes, dFF, 'k')
        ax.set_xlabel('number of spikes')
        ax.set_ylabel('peak dFF')
        
        # Return figure
        return fig
    
    def select_nodes(self, timeseries, inodes='all'):
        '''
        Select nodes from simulation output dataframe.

        :param timeseries: multi-indexed simulation output dataframe
        :param inodes: list of nodes to select (default: 'all'). One of:
            - 'all' (for all nodes)
            - 'stim' (for sonictaed nodes only)
            - 'nostim' (for non-sonicated nodes only)
            - list of node indexes
        :return: selected nodes list
        '''
        # If inodes is a string, generate corresponding list of node indexes
        if isinstance(inodes, str):
            # If all nodes requested, return to all nodes
            if inodes == 'all':
                return self.nodelist
            # Otherwise, compute max stimulus amplitude per node
            else:
                Amax = timeseries[self.STIM_KEY].groupby(self.NODE_KEY).max()
                # If only stimulated nodes requested, extract nodes with non-null max amplitude
                if inodes == 'stim':
                    return Amax[Amax > 0].index.values
                # If only non-stimulated nodes requested, extract nodes with null max amplitude
                elif inodes == 'nostim':
                    return Amax[Amax == 0].index.values
        # If inodes is an integer or a list of integers, convert to list of node names
        else:
            inodes = as_iterable(inodes)
            if isinstance(inodes[0], int):
                return [self.nodelist[i] for i in inodes]

        # Otherwise, return inodes as-is
        return inodes

    def compute_metric(self, data, metric, inodes='all'):
        '''
        Compute metric across nodes from simulation output.
        
        :param data: multi-indexed simulation output dataframe, 
            and potential presynaptic drive events
        :param metric: metric name
        :param inodes: list of nodes to compute metric for (default: 'all'). One of:
            - 'all' (for all nodes)
            - 'stim' (for sonictaed nodes only)
            - 'nostim' (for non-sonicated nodes only)
            - list of node indexes
        :return: array(s) of metric value per node
        '''
        # If multiple metrics requested, call function for each of them
        if is_iterable(metric):
            mdict = {m: self.compute_metric(data, m, inodes=inodes) for m in metric}
            mdict = {m: v.to_frame() if isinstance(v, pd.Series) else v for m, v in mdict.items()}
            return pd.concat(mdict, axis=1)
        
        # Unpack data
        if isinstance(data, tuple):
            timeseries, events = data
        else:
            timeseries, events = data, None

        # If input data has extra dimensions, group by these extra dims
        # and compute metric across groups
        gby = [k for k in timeseries.index.names if k not in (self.NODE_KEY, self.TIME_KEY)]
        if len(gby) > 0:
            if len(gby) == 1:
                gby = gby[0]
            mdict = {}
            for gkey, gdata in timeseries.groupby(gby):
                gdata = gdata.droplevel(gby)
                if events is not None:
                    gdata = (gdata, events.loc[gkey])
                mdict[gkey] = self.compute_metric(gdata, metric, inodes=inodes)
            return self.concatenate_outputs(mdict.keys(), list(mdict.values()), gby)
        
        # If inodes is a string, generate corresponding list of node indexes
        inodes = self.select_nodes(timeseries, inodes=inodes)

        # Filter timeseries and events according to requested nodes
        mux_slice = [slice(None)] * timeseries.index.nlevels
        mux_slice[timeseries.index.names.index(self.NODE_KEY)] = inodes
        timeseries = timeseries.loc[tuple(mux_slice)]
        if events is not None:
            mux_slice = [slice(None)] * events.index.nlevels
            mux_slice[events.index.names.index(self.NODE_KEY)] = inodes
            events = events.loc[tuple(mux_slice)]

        # Extract time vector from timeseries
        t = self.extract_time(timeseries) # ms

        # Extract metric from timeseries
        if metric == 'nspikes':
            m = self.extract_ap_counts(t, self.extract_var(timeseries, 'v'))
        elif metric =='nspikes_evoked':
            if events is None:
                raise ValueError('cannot compute nspikes_evoked: no pre-synaptic drive events provided')
            tspikes = self.extract_ap_times(t, self.extract_var(timeseries, 'v'))
            is_artificial = self.detect_artificial_spikes(tspikes, events).to_dict()
            m = np.array([np.sum(~v) for v in is_artificial.values()])
        elif metric == 'ΔT':
            T = self.extract_var(timeseries, 'T')  # °C
            m = np.max(T, axis=1) - np.min(T, axis=1)
        elif metric == 'FR':
            m = self.extract_firing_rates(t, self.extract_var(timeseries, 'v'))
        elif metric == 'ΔFR/FRpresyn':
            if not self.is_drive_set():
                raise ValueError('cannot compute FR/FRpresyn: no pre-synaptic drive set')
            freq = self.get_drive_params()['freq']
            m = self.extract_firing_rates(t, self.extract_var(timeseries, 'v'))
            m = tuple((np.asarray(mm) - freq) / freq for mm in m)
        elif metric == self.RESP_KEY:
            m = self.extract_stim_response(timeseries)
        elif metric == 'charge':
            m = self.compute_charge(timeseries)
        else:
            raise ValueError(f'Invalid metric: {metric}')

        # If metric is a tuple, convert to mean - err dataframe  
        if isinstance(m, tuple):
            m = pd.DataFrame({'mean': m[0], 'err': m[1]}).add_suffix(f' {metric}')

        # If metric is array/list, convert to series
        elif isinstance(m, (np.ndarray, list)):
            m = pd.Series(m, name=metric)

        # If metric has not specific index, define it
        if m.index.name is None:
           m.index = pd.Index(inodes, name=self.NODE_KEY)

        # Return
        return m
    
    def filter_timeseries(self, timeseries, **kwargs):
        '''
        Filter simulation output timeseries according to inclusion and exclusion criteria.

        :param timeseries: multi-indexed simulation output timeseries dataframe
        :param kwargs: inclusion and exclusion criteria
        :return: filtered output timeseries dataframe
        '''
        # Extract list of keys from inclusion and exclusion criteria
        keys = self.filter_record_keys(**kwargs)

        # If stimulus is set, add stimulus key to list of keys
        if self.is_stim_set():
            keys.append(self.STIM_KEY)

        # Get intersection between keys and output timeseries keys 
        # (in case some currents are disabled)
        keys = [k for k in keys if k in timeseries.keys()]

        # Return filtered output dataframe
        return timeseries[keys]
    
    def integrate_current(self, i):
        '''
        Integrate current density vector

        :param i: current density vector (in mA/cm2) provided as a time (ms)-indexed pandas series 
        :return: charge density (in uC/cm2)
        '''
        if not isinstance(i, pd.Series) or i.index.name != self.TIME_KEY:
            raise ValueError(f'input current must be a time-indexed series')
        return integrate.trapezoid(i.values, x=i.index.values)  # mA/cm2 * ms = uC/cm2
    
    def compute_charge(self, data, tbounds=None):
        '''
        Compute charge injected by each current within a time interval 

        :param data: multi-indexed simulation output dataframe (and potential presynaptic drive events)
        :param tbounds: time interval to consider to charge computation: either a tuple of time values, or "half2"  
        :return: dataframe of accumulated charges per current
        '''
        # Unpack input data
        if isinstance(data, tuple):
            timeseries, _ = data
        else:
            timeseries = data

        # If time bounds provided, filter timeseries
        if tbounds is not None:
            # If tbounds is a string code, convert to float interval
            if isinstance(tbounds, str):
                if tbounds.startswith('stim'):
                    tb = [self.start, self.start + self.dur]
                    if tbounds == 'stimhalf1':
                        tb[1] = self.start + self.dur / 2
                    elif tbounds == 'stimhalf2':
                        tb[0] = self.start + self.dur / 2
                    tbounds = tb
                else: 
                    raise ValueError(f'invalid tbounds code: {tbounds}')
            tidx = timeseries.index.get_level_values(self.TIME_KEY)
            isin = np.logical_and(tidx >= tbounds[0], tidx <= tbounds[1])
            timeseries = timeseries[isin]
        
        # Check that currents are in timeseries data
        hascurrs = 'currents' in self.record_dict and any(k in timeseries for k in self.record_dict['currents'])
        if not hascurrs: 
            raise ValueError('no current found in input data')
        
        # Initialize charges dictionary
        charges = {}

        # Identify non-time keys in timeseries
        gby = [k for k in timeseries.index.names if k != self.TIME_KEY]
        
        # Loop through currents and integrate to compute charges
        for ckey in self.record_dict['currents']:
            if ckey in timeseries and ckey != 'iNoise':
                charges[ckey] = (
                    timeseries[ckey]
                    .groupby(gby)
                    .agg(lambda i: self.integrate_current(i.droplevel(gby)))
                )
        
        # Assemble into dataframe
        charges = pd.concat(charges, axis=1, names='current')

        # Return charges
        return charges
    
    def compute_charge_by_stimhalf(self, data, **kwargs):
        ''' Compute injected charges by each current in first and second half of stimulus '''
        thalfs = {
            1: 'stimhalf1',
            2: 'stimhalf2'
        }
        charges = pd.concat(
            {k: self.compute_charge(data, tbounds=v, **kwargs) for k, v in thalfs.items()},
            axis=0,
            names=['stim. half']
        )
        return charges

    def plot_results(self, data, tref='onset', gmode='abs', addstimspan=True, title=None, 
                     rectify_currents=False, clip_currents=False, add_net_current=False, 
                     add_spikes=False, hrow=1.5, wcol=2, lw=1, **kwargs):
        '''
        Plot the time course of variables recorded during simulation.
        
        :param data: multi-indexed simulation output dataframe (and potential presynaptic drive events)
        :param tref: time reference for x-axis (default: 'onset')
        :param gmode: conductance plotting mode (default: 'abs'). One of:
            - "abs" (for absolute)
            - "rel" (for relative)
            - "norm" (for normalized)
            - "log" (for logarithmic) 
        :param addstimspan: whether to add a stimulus span on all axes (default: True)
        :param title: optional figure title (default: None)
        :param rectify_currents: whether to rectify currents so that they all have 
            the same polarity on the plot (default: False)
        :param clip_currents: whether to clip currents with large transient amplitudes to better
            appreciate the dynamics of other currents (default: False)
        :param add_net_current: whether to add a line of the net membrane current 
            on the currents graph (default: False)
        :return: figure handle 
        '''
        # Check that plotting mode is valid
        if gmode not in ['abs', 'rel', 'log', 'norm']:
            raise ValueError(f'Invalid conductance plotting mode: {gmode}')

        # Unpack input data
        events = None
        if isinstance(data, tuple):
            timeseries, events = data
        else:
            timeseries = data

        # Filter output 
        timeseries = self.filter_timeseries(timeseries, **kwargs)

        # Log
        self.log('plotting results')

        # Assess which variable types are present in timeseries
        hasstim = self.STIM_KEY in timeseries
        hastemps = 'T' in timeseries
        hasconds = 'conductances' in self.record_dict and any(k in timeseries for k in self.record_dict['conductances'])
        hascurrs = 'currents' in self.record_dict and any(k in timeseries for k in self.record_dict['currents'])
        hasvoltages = 'v' in timeseries

        # Check that number of nodes in output corresponds to model size
        nnodes_out = len(timeseries.index.unique(self.NODE_KEY))
        assert nnodes_out == self.size, f'number of nodes ({self.size}) does not match number of output voltage traces ({nnodes_out})'
        colkey = self.NODE_KEY
        dropkey = colkey
        
        # if only 1 node, check if extra dimension can be used for columns
        if nnodes_out == 1:
            extra_dims = [k for k in timeseries.index.names if k not in [self.NODE_KEY, self.TIME_KEY]]
            if len(extra_dims) > 0:
                colkey = extra_dims[0]
                dropkey = [colkey, self.NODE_KEY]

        # Create figure with appropriate number of rows and columns
        nrows = int(hasstim) + int(hastemps) + int(hasconds) + int(hascurrs) + int(hasvoltages)
        
        ncols = len(timeseries.index.unique(colkey))
        fig, axes = plt.subplots(
            nrows, 
            ncols, 
            figsize=(wcol * ncols + 1.5, hrow * nrows), 
            sharex=True, 
            sharey='row'
        )
        if ncols == 1:
            axes = np.atleast_2d(axes).T
        sns.despine(fig=fig)
        for ax in axes[:-1].flatten():
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for ax in axes[:, 1:].flatten():
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        for ax in axes[:, 0]:
            ax.spines['left'].set_position(('outward', 5))
        for ax in axes[-1]:
            ax.set_xlabel(self.TIME_KEY)
        for (v, _), ax in zip(data.groupby(colkey), axes[0]):
            ax.set_title(f'{colkey} = {v}')

        # Define legend keyword arguments
        leg_kwargs = dict(
            bbox_to_anchor=(1.0, .5),
            loc='center left',
            frameon=False,
        )

        # If stimulus timeseries data exists
        if hasstim:
            # Extract max stimulus amplitude per column, and complete column titles
            Pamps = timeseries[self.STIM_KEY].groupby(colkey).max()
            if colkey == self.NODE_KEY:
                for ax, Pamp in zip(axes[0], Pamps):
                    ax.set_title(f'{ax.get_title()}: {Pamp:.1f} MPa')
            
            # Extract stimulus bounds            
            tvec, _ = self.get_stim_vecs()
            stimbounds = np.array([tvec[1], tvec[-2]])
        
            # If specified, offset time to align 0 with stim onset
            if tref == 'onset':
                mux_df = timeseries.index.to_frame()
                mux_df[self.TIME_KEY] -= stimbounds[0]
                timeseries.index = pd.MultiIndex.from_frame(mux_df)
                if events is not None:
                    events = events - stimbounds[0]
                stimbounds -= stimbounds[0]
 
            # If specified, mark stimulus span on all axes
            if addstimspan:
                for irow, axrow in enumerate(axes):
                    for ax in axrow:
                        ax.axvspan(
                            *stimbounds, fc='silver', ec=None, alpha=.3,
                            label='bounds' if irow == 0 else None)
            
            # Set x-axis ticks to stimulus bounds
            for ax in axes[-1]:
                ax.set_xticks(stimbounds)
        
        # Initialize axis row index
        irow = 0

        # Plot stimulus time-course per column
        if hasstim:
            axrow = axes[irow]
            axrow[0].set_ylabel(self.P_MPA_KEY)
            for ax, (_, stim) in zip(axrow, timeseries[self.STIM_KEY].groupby(colkey)):
                stim.droplevel(dropkey).plot(ax=ax, c='k', lw=lw)
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Plot temperature time-course per column
        if hastemps:
            axrow = axes[irow]
            axrow[0].set_ylabel('T (°C)')
            for ax, (_, T) in zip(axrow, timeseries['T'].groupby(colkey)):
                T.droplevel(dropkey).plot(ax=ax, c='k', lw=lw)
            irow += 1

        # For each channel type, plot conductance time course per column
        if hasconds:
            axrow = axes[irow]
            if gmode == 'rel':
                ylabel = 'g/g0 (%)'
            elif gmode == 'norm':
                ylabel = 'g/gmax (%)'
            else:
                ylabel = 'g (uS/cm2)'
            axrow[0].set_ylabel(ylabel)
            for condkey, color in zip(self.record_dict['conductances'], plt.get_cmap('Dark2').colors):
                if condkey in timeseries:
                    for ax, (_, g) in zip(axrow, timeseries[condkey].groupby(colkey)):
                        g = g.droplevel(dropkey)
                        if condkey.endswith('bar'):
                            label = f'\overline{{g_{{{condkey[1:-3]}}}}}'
                        else:
                            label = f'g_{{{condkey[1:]}}}'
                        if gmode == 'rel':
                            if g.iloc[0] == 0:
                                self.log(f'Cannot compute relative conductance for {label}: baseline is 0', warn=True)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                g = g / g.iloc[0] * 100
                        elif gmode == 'norm':
                            g = g / g.max() * 100
                        g.plot(ax=ax, label=f'${label}$', color=color, lw=lw)
            axrow[-1].legend(**leg_kwargs)
            if gmode == 'log':
                for ax in axrow:
                    ax.set_yscale('log')
            irow += 1

        # For each channel type, plot current time course per column
        if hascurrs:
            bounds_per_ax = np.zeros((len(axes[irow]), 2))
            ncurrs = 0
            axrow = axes[irow]
            clabel = '|I|' if rectify_currents else 'I'
            axrow[0].set_ylabel(f'{clabel} (uA/cm2)')
            zorders = dict(zip(
                self.record_dict['currents'].keys(),
                np.arange(len(self.record_dict['currents'].keys()))[::-1]
            ))
            for ckey in self.record_dict['currents']:
                if ckey in timeseries and ckey != 'iNoise':
                    color = self.CURRENTS_CMAP[ckey]
                    # Determine sign switch for current
                    sign = 1
                    if rectify_currents and ckey in self.RECTIFIED_CURRENTS:
                        sign = -1
                    
                    # Determine alpha value for current
                    alpha = 1.
                    # if clip_currents and ckey in self.CLIPPED_CURRENTS:
                    #     alpha = 0.5
                    
                    # Plot current time course per column
                    for iax, (ax, (_, current)) in enumerate(zip(axrow, timeseries[ckey].groupby(colkey))):
                        s = sign * current.droplevel(dropkey) * 1e3  # uA/cm2
                        s.plot(
                            ax=ax, label=f'$i_{{{ckey[1:]}}}$', color=color, alpha=alpha, zorder=zorders[ckey], lw=lw)
                        if ckey not in self.CLIPPED_CURRENTS:
                            bounds_per_ax[iax, 0] = min(bounds_per_ax[iax, 0], s.min())
                            bounds_per_ax[iax, 1] = max(bounds_per_ax[iax, 1], s.max())
                    ncurrs += 1
            
            # If net current requested, plot it on top
            if add_net_current:
                for ax, (_, idata) in zip(axrow, timeseries['iNet'].groupby(colkey)):
                    s = idata.droplevel(dropkey) * 1e3  # uA/cm2
                    s.plot(ax=ax, label='$i_{net}$', color='k', ls='--', zorder=30, lw=lw)
                ncurrs += 1
            
            # If currents clipping requested
            if clip_currents:
                # Compute min and max across axes
                ybounds = np.array([bounds_per_ax[:, 0].min(), bounds_per_ax[:, 1].max()])
                yrange = np.diff(ybounds)[0]

                # If y range is non-zero
                if yrange > 0:
                    # Compute symmetric y bounds if y range is on both sides of 0
                    if ybounds[0] * ybounds[1] < 0:
                        ybounds = np.array([-max(np.abs(ybounds)), max(np.abs(ybounds))])

                    # Add relative margin to y-axis bounds for clipping
                    ybounds = expand_range(ybounds, factor=0.1)

                    # Loop through axes
                    for ax in axrow:
                        # Set dictionary of clipping status for any trace
                        is_any_clipped = {'lb': False, 'ub': False}
                        
                        # Loop through current traces
                        for line in ax.lines:
                            # Clip current trace while adding new data points 
                            # at locations where the trace crosses the restricted y bounds
                            # (to preserve the trace appearance within the y range)
                            xdata, ydata, is_clipped = restrict_trace(
                                line.get_xdata(), line.get_ydata(), *ybounds, 
                                full_output=True)
                            line.set_data(xdata, ydata)

                            # Update clipping status
                            for k, v in is_clipped.items():
                                is_any_clipped[k] |= v
                        
                        # Add dash horizontal lines at clipped y bounds
                        for (k, v), yb in zip(is_any_clipped.items(), ybounds):
                            if v:
                                ax.axhline(yb, c='k', ls='--', lw=0.5)

                    # Set y-axis clipping bounds with extra margin
                    exp_ybounds = expand_range(ybounds, factor=0.1)
                    # Set new y limits to first axis (sharey should adjust other axes)
                    axrow[0].set_ylim(*exp_ybounds)
            
            axrow[-1].legend(ncols=int(np.ceil(ncurrs / 4)), **leg_kwargs)
            irow += 1

        # Plot membrane potential time-course per column
        if hasvoltages:
            axrow = axes[irow]
            axrow[0].set_ylabel('Vm (mV)')
            for ax, (k, v) in zip(axrow, timeseries['v'].groupby(colkey)):
                v = v.droplevel(dropkey)
                v.plot(ax=ax, c='k', lw=lw)
                
                if add_spikes:
                    # Detect spikes timings
                    aptimes = self.extract_ap_times(v.index, v.values)
                    # If pre-synaptic drive events
                    if events is not None:
                        # Plot pre-synaptic drive events raster
                        self.add_events_raster(
                            ax, events.loc[k], y=85, color='dimgray', label='drive evts.')                    
                        # Classify spikes as "artificial" or "real"
                        is_artificial = self.detect_artificial_spikes(aptimes, events.loc[k])
                        # Plot raster for each spike type
                        for is_art, color, label in zip([True, False], ['k', 'g'], ['art. spikes', 'evoked spikes']):
                            ts = aptimes[is_artificial == is_art]
                            if len(ts) > 0:
                                self.add_events_raster(ax, ts, y=70, color=color, label=label)                

                    # Otherwise, plot all spikes
                    else:
                        self.add_events_raster(
                            ax, aptimes, y=70, color='k', label='spikes')

                # Force minimal y-axis limits
                yb, yt = ax.get_ylim()
                ax.set_ylim(min(yb, -80), max(yt, 50))
            
            if add_spikes:
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
    
    def add_events_raster(self, ax, times, symbol='|', color='k', y=None, dy=0.05, label=None, marker_size=5):
        ''' 
        Add a raster of events to an axis.

        :param ax: axis handle
        :param times: event times (ms)
        :param symbol: symbol to use for events (default: '*')
        :param color: color to use for events (default: 'k')
        :param y: absolute y position of events (default: None). If None, the y position 
            is computed dynamically from the axis y range.
        :param dy: relative y offset of events w.r.t. axis y range (default: 0.1)
        '''
        # Determine y position of raster symbols, if not provided
        if y is None:
            ylims = ax.get_ylim()
            yext = ylims[1] - ylims[0]
            y = ylims[1] + yext * dy
        # Plot raster symbols
        ax.plot(times, np.full_like(times, y), symbol, color=color, label=label, ms=marker_size)
    
    def get_stimdist_vector(self, kind='all'):
        '''
        Return stimulus distribution vector for a given kind of stimulus distribution.

        :param kind: stimulus distribution kind (default: 'all'). One of:
            - "single": single node stimulated
            - "all": all nodes stimulated with equal amplitude
            - integer/float: custom fraction of nodes stimulated with equal amplitude
        :return: stimulus distribution vector
        '''        
        # Return stimulus distribution vector
        if kind == 'single':
            x = np.zeros(self.size)
            x[0] = 1
            return x
        elif kind == 'all':
            return np.ones(self.size)
        elif isinstance(kind, (int, float)):
            frac = kind
            if frac < 0 or frac > 1:
                raise ValueError(f'Invalid stimulus distribution fraction: {frac}')
            inodes = np.random.choice(self.inodes, int(frac * self.size), replace=False)
            x = np.zeros(self.size)
            x[inodes] = 1.
            return x 
        else:
            raise ValueError(f'Invalid stimulus distribution kind: {kind}')
    
    def get_stimdists(self):
        '''
        Return dictionary of stimulus distribution vectors for all stimulus distribution kinds.
        '''
        return {kind: self.get_stimdist_vector(kind) for kind in ['single', 'all']}
        
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
    
    def run_stim_sweep(self, amps, stimdist=None, nreps=1, **kwargs):
        ''' 
        Simulate model across a range of stimulus amplitudes and return outputs.
        
        :param Pamps: range of stimulus amplitudes
        :param stimdist (optional): vector spcifying relative stimulus amplitudes at each node. 
            If not provided, all nodes will be stimulated with the same amplitude.
        :param nreps (optional): number of repetitions per stimulus amplitude (default: 1)
        :param kwargs: optional arguments passed to "set_stim" and "simulate" methods
        :return: multi-indexed output dataframe with sweep variable, node index and time
        '''
        # If stimulus distribution vector not provided, assume uniform distribution across nodes
        if stimdist is None:
            stimdist = [1] * self.size
        
        # Check that stimulus distribution vector is valid
        self.check_stimulus_distribution(stimdist)
        self.log(f'running simulation sweep across {len(amps)} stimulus amplitudes')

        # Generate 2D array of stimulus vectors for each stimulus amplitude
        amp_vec_range = np.dot(np.atleast_2d(amps).T, np.atleast_2d(stimdist))

        # Disable verbosity during sweep
        vb = self.verbose
        self.verbose = False

        # Initialize empty data list
        data = []

        # Simulate model for each stimulus vector, and append output to data
        tstop = kwargs.pop('tstop', None)
        dt = kwargs.pop('dt', None)
        for amp_vec in tqdm(amp_vec_range):
            self.set_stim(amp_vec, **kwargs)
            data.append(self.simulate(tstop=tstop, nreps=nreps, dt=dt))
        
        # Restore verbosity
        self.verbose = vb

        # Concatenate data and events with new sweep index level, and return
        return self.concatenate_outputs(amps, data, self.P_MPA_KEY)
    
    def plot_sweep_results(self, data, metric=None, title=None, ax=None, width=4, height=2, 
                           legend=True, marker='o', markersize=4, lw=1, 
                           estimator='mean', errorbar='se', Pmap='full', xscale='linear'):
        '''
        Plot results of a sweep.
        
        :param data: multi-indexed output dataframe with sweep variable and node index.
        :param metric (optional): metric(s) to plot
        :param title: optional figure title (default: None)
        :param ax: optional axis handle (default: None)
        :param width: figure width (default: 4)
        :param height: figure height (default: 2)
        :param legend: whether to add a legend to the graph(s)
        :param estimator: estimator for the central tendency (default: 'mean').
        :param errorbar: estimator for the error bars (default: 'se')
        :param Pmap: whether/how to add a pressure mapping to the graph(s). On of:
            - False: no mapping
            - "ticks": only ticks
            - "full": ticks, tick labels and axis label
        :param xscale: scale for the x-axis (default: 'linear'). One of:
            - "linear"
            - "log"
            - "symlog"
            - "sqrt"
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
                    self.plot_sweep_results(
                        data, m, ax=ax, legend=legend, 
                        estimator=estimator, errorbar=errorbar, Pmap=Pmap, xscale=xscale)
                    if Pmap == 'full':
                        Pmap = 'ticks'
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

        # If data contains raw simulation output, compute metric
        if isinstance(data, tuple) or self.TIME_KEY in data.index.names:
            data = self.compute_metric(data, metric)

        # Extract index dimensions other than "node" and "repetition"
        extradims = [k for k in data.index.names if k not in (self.NODE_KEY, self.REP_KEY)]
        
        # If no extra dimension, raise error
        if len(extradims) == 0:
            raise ValueError('Cannot plot sweep results with no extra grouping dimension')
        # If 1 extra dimension, extract sweep key and set extra dimension key to None
        elif len(extradims) == 1:
            sweepkey, extrakey = extradims[0], None
        # If 2 extra dimensions, extract sweep key and extra dimension key
        if len(extradims) == 2:
            try:
                idx = extradims.index(self.P_MPA_KEY)
                sweepkey, extrakey = extradims[idx], extradims[1 - idx]
            except ValueError:
                extrakey, sweepkey = extradims
        # If too many extra dimensions, raise error
        elif len(extradims) > 2:
            raise ValueError('Cannot plot sweep results with more than 2 extra grouping dimensions')

        # If relative change, convert to percentage
        if metric == self.RESP_KEY:
            data = data * 100

        # If metric data is dataframe, extract mean column and store dataframe
        mvar = None
        if isinstance(data, pd.DataFrame):
            # Identify metric name from common suffix in all columns
            if metric is None:
                metric = os.path.commonprefix([c[::-1] for c in data.columns])[::-1].strip()
            data, mvar = data[f'mean {metric}'].rename(metric), data
        else:
            if metric is None:
                metric = data.name
        
        # Generate color palette
        colors = list(plt.get_cmap('tab10').colors)
        nodes = data.index.get_level_values(self.NODE_KEY).unique()
        palette = dict(zip(nodes, colors))

        # Plot metric vs sweep variable
        sns.lineplot(
            data=data.to_frame(), 
            ax=ax, 
            x=sweepkey,
            y=metric,
            hue=self.NODE_KEY if len(nodes) > 1 else None,
            marker=marker,
            markersize=markersize,
            linewidth=lw,
            palette=palette if len(nodes) > 1 else None,
            estimator=estimator,
            errorbar=errorbar,
            style=extrakey,
            legend=legend
        )

        # If sweep variance is provided
        if mvar is not None:
            # Compute lower and upper bounds from mean and err columns
            mvar[f'lb {metric}'] = mvar[f'mean {metric}'] - mvar[f'err {metric}']
            mvar[f'ub {metric}'] = mvar[f'mean {metric}'] + mvar[f'err {metric}']
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
                    gdata.droplevel(gby).index, gdata[f'lb {metric}'], gdata[f'ub {metric}'], 
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
        
        # Set x-axis scale
        self.set_ax_scale(ax, key='x', scale=xscale)

        # Add title, if specified
        if title is not None:
            ax.set_title(title)
        
        # Return figure
        return fig

    def compute_istim(self, Pamp):
        ''' 
        Compute stimulus-driven current amplitude for a given stimulus amplitude

        :param Pamp: peak pressure amplitude (MPa)
        :return: stimulus-driven current amplitude (mA/cm2)
        '''
        # Compute stimulus-driven current amplitude over amplitude range
        iStim = -self.a * np.power(Pamp)

        # If input is an array, format as series
        if isinstance(Pamp, (tuple, list, np.ndarray)):
            iStim = pd.Series(iStim, index=Pamp, name='iStim (mA/cm2)')
            iStim.index.name = self.P_MPA_KEY
            iStim -= iStim.max()

        # Return
        return iStim
    
    def compute_Tmax(self, Pamp, dur=None):
        '''
        Compute temperature reached at the end of a stimulus of 
        specific amplitude and duration.

        :param Pamp: peak pressure amplitude (MPa)
        :param dur (optional): stimulus duration (ms)
        :return: maximal temperature reached (°C)
        '''
        # If input is an list/tuple, convert to numpy array
        if isinstance(Pamp, (tuple, list)):
            Pamp = np.asarray(Pamp)

        # If duration not provided, use class attribute
        if dur is None:
            if self.dur is None:
                raise ValueError('No stimulus duration defined')
            dur = self.dur

        # Compute acoustic intensity during stimulus
        Isppa = pressure_to_intensity(Pamp * 1e6) * 1e-4  # W/cm2 
        
        # Compute steady-state temperature
        ΔTinf = self.alphaT * Isppa
        Tinf = ΔTinf + self.Tref

        # Compute temperature incrase at the end of the stimulus
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            c1 = -self.tauT_abs * np.log((ΔTinf))
        Tmax = Tinf - np.exp(-(dur + c1) / self.tauT_abs)

        # If Pamp or duration is an array, format as series
        if isinstance(Pamp, (tuple, list, np.ndarray)):
            Tmax = pd.Series(Tmax, index=Pamp, name='Tmax (°C)')
            Tmax.index.name = self.P_MPA_KEY
        if isinstance(dur, (tuple, list, np.ndarray)):
            Tmax = pd.Series(Tmax, index=dur, name='Tmax (°C)')
            Tmax.index.name = self.DUR_KEY

        # Return
        return Tmax
    
    def compute_iKTmax(self, Pamp, **kwargs):
        '''
        Compute maximal thermally-activated potassium current amplitude
        for a given stimulus amplitude

        :param Pamp: peak pressure amplitude (MPa)
        :return: thermally-activated potassium current amplitude (mA/cm2)
        '''
        # Compute maximal temperature reached
        Tmax = self.compute_Tmax(Pamp, **kwargs)
        
        # Compute potassium current for that temperature
        gKTmax = self.gKT * (Tmax - self.Tref)
        iKTmax = gKTmax * (self.vrest - self.EKT)

        # If Series input, rename
        if isinstance(iKTmax, pd.Series):
            iKTmax.name = 'iKTmax (mA/cm2)'

        # Return
        return iKTmax
    
    def compute_EI_currents(self, Pamp, dur=None):
        '''
        Compute excitatory and inhibitory currents for a given stimulus amplitude

        :param Pamp: peak pressure amplitude (MPa)
        :return: dataframe of excitatory and inhibitory currents (mA/cm2)
        '''
        # If duration not provided, use class attribute
        if dur is None:
            if self.dur is None:
                raise ValueError('No stimulus duration defined')
            dur = self.dur

        # Compute stimulus-driven and thermally-activated currents amplitude
        # over Pamp range
        return pd.concat([
            self.compute_istim(Pamp),  # stimulus-driven current
            self.compute_iKTmax(Pamp, dur=dur),  # thermally-activated current at end of stimulus
            self.compute_iKTmax(Pamp, dur=dur / 2).rename('iKT1/2 (mA/cm2)'),  # thermally-activated current at half stimulus duration
        ], axis=1)
    
    def plot_EI_imbalance(self, Pamp, ax=None, add_Pmap=True, legend=True, ls='-', 
                          xscale='linear', **kwargs):
        '''
        Plot the imbalance between excitatory and inhibitory currents 
        over a range of stimulus amplitudes.

        :param Pamp: peak pressure amplitude (MPa) or range of amplitudes
        :param ax: optional axis handle (default: None)
        :param add_Pmap: whether to add x-axis with pressure values mapped to input Isspa values (default: False)
        :param legend: whether to add a legend to the graph(s)
        :param ls: line style (default: '-')
        :param xscale: scale for the x-axis (default: 'linear'). One of:
            - "linear"
            - "log"
            - "symlog"
            - "sqrt"
        :return: figure handle
        '''
        # If Pamp is a scalar of a size 1 vector, generate dense linear range between 0 and that value
        if not isinstance(Pamp, (tuple, list, np.ndarray)) or len(Pamp) < 2:
            Pamp = np.linspace(0, Pamp, 100)
        
        # Compute excitatory and inhibitory currents over pressure amplitude range, and 
        # convert to absolute values
        df = self.compute_EI_currents(Pamp, **kwargs).abs()

        # Remove units from currents names
        df.columns = df.columns.str.rstrip('(mA/cm2)').str.rstrip(' ')

        # Convert to uA/cm2 
        df = df * 1e3

        # Create / retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Plot currents
        ax.set_ylabel('|I| (uA/cm2)')
        colors = list(plt.get_cmap('tab20').colors)
        palette = {
            'iStim': colors[0],
            'iKTmax': colors[2],
            'iKT1/2': colors[3],
        }
        for k in df.columns:
            df[k].plot(ax=ax, c=palette[k], ls=ls, label=k)
        if legend:
            ax.legend(bbox_to_anchor=(1.0, .5), loc='center left', frameon=False)

        # Set x-axis scale
        self.set_ax_scale(ax, key='x', scale=xscale)

        # Add pressure axis, if requested
        if add_Pmap:
            self.add_pressure_mapping(ax)

        # Return
        return fig
    
    @staticmethod
    def get_data_bounds(axbounds, alpha=0.05, scale='linear'):
        ''' 
        Extract data bounds from axis bounds
        
        :param axbounds: axis bounds
        :param alpha: fraction of axis range to remove from both ends (default: 0.05)
        :return: data bounds
        '''
        # If scale is square root, convert axis bounds to square space
        if scale == 'sqrt':
            axbounds = np.square(axbounds) * np.sign(axbounds)
        # Compute axis range
        axdx = axbounds[1] - axbounds[0]
        # Infer data range
        dx = axdx / (1 + 2 * alpha)
        # Compute data bounds 
        bounds = np.array([axbounds[0] + dx * alpha, axbounds[1] - dx * alpha])
        # If scale is square root, convert data bounds back to original space
        if scale == 'sqrt':
            bounds = np.sqrt(bounds)
        # Return
        return bounds
    
    @staticmethod
    def get_axis_bounds(bounds, alpha=0.05, scale='linear'):
        '''
        Extract axis bounds from data bounds

        :param bounds: data bounds
        :param alpha: fraction of axis range to add on both ends (default: 0.05)
        :return: axis bounds
        '''
        # If scale is square root, convert data bounds to sqrt space
        if scale == 'sqrt':
            bounds = np.sqrt(bounds)
        # Compute data range
        dx = bounds[1] - bounds[0]
        # Add margin to set axis bounds
        axbounds = np.array([bounds[0] - dx * alpha, bounds[1] + dx * alpha])
        # If scale is square root, convert axis bounds back to original space
        if scale == 'sqrt':
            axbounds = np.square(axbounds) * np.sign(axbounds)
        # Return
        return axbounds

    @classmethod
    def set_ax_scale(cls, ax, key='x', scale='linear'):
        ''' 
        Set x or y axis scale of an axis object to a particular projection
        
        :param ax: axis handle
        :param key: axis key (default: 'x'). One of "x", "y", or "xy"
        :param scale: scale for the x-axis (default: 'linear'). One of:
            - "linear"
            - "log"
            - "symlog"
            - "sqrt"
        '''
        if scale is None:
            return
        
        # If multiple keys, recursively call function for each key
        if len(key) > 1:
            for k in key:
                cls.set_ax_scale(ax, key=k, scale=scale)
            return

        # Check that scale is valid, and extract scale function
        if key == 'x':
            setscalefunc = ax.set_xscale
            getlimfunc, setlimfunc = ax.get_xlim, ax.set_xlim
        elif key == 'y':
            setscalefunc = ax.set_yscale
            getlimfunc, setlimfunc = ax.get_ylim, ax.set_ylim
        else:
            raise ValueError(f'Invalid axis key: {key}')
        
        # If scale if square root
        if scale == 'sqrt':
            # Extract axis bounds (does nothing, but otherwise does not work)
            _ = getlimfunc()
            # Set scale function to square root
            setscalefunc('function', functions=(signed_sqrt, np.square))
            # Adjust axis bounds to ensure even margin in axis space
            axbounds = getlimfunc()
            bounds = cls.get_data_bounds(axbounds)
            bounds = np.max((bounds, np.zeros(2)), axis=0)
            axbounds = cls.get_axis_bounds(bounds, scale='sqrt')
            setlimfunc(axbounds)
        # Otherwise, set scale function to requested scale
        else:
            setscalefunc(scale)
    
    def run_comparative_sweep(self, Pamps, metric, stimdists=None, sigmaI=None, **kwargs):
        '''
        Compute metric over Pamp sweep for both single node and uniform stimulus distributions,

        :param Pamps: peak pressure amplitude vector (MPa)
        :param metric: name of metric to compute
        :param stimdists (optional): stimulus distribution vector(s) (default: None). If not provided,
            both single node and uniform distributions will be used.
        :return: multi-indexed metric dataframe
        '''
        # If stimulus distribution vectors not provided, use default ones
        if stimdists is None:
            stimdists = self.get_stimdists()

        # Initialize lists of stimulus distribution keys and metric results
        mdata, kinds = [], []
        
        # For each stimulus distribution
        for kind, dist in stimdists.items():
            # If required, add random variations in relative stimulus amplitudes
            if sigmaI is not None:
                dist = np.array([max(np.random.normal(I, sigmaI * I), 0) for I in dist])
            # Run sweep over stimulus amplitudes 
            data = self.run_stim_sweep(Pamps, stimdist=dist, **kwargs)
            # Compute metric, and append to list
            mdata.append(self.compute_metric(data, metric))
            # Append stimulus distribution key to list
            kinds.append(kind)

        # Concatenate metric results and return
        return self.concatenate_outputs(kinds, mdata, 'stim dist.')

    def explore2D(self, wrange, arange, Pamp_range, title=None, metric='nspikes', **kwargs):
        '''
        Explore 2D parameter space of synaptic weight and stimulus sensitivity.
        
        :param wrange: synaptic weight range (S/cm2)
        :param arange: stimulus sensitivity range (-)
        :param Pamp_range: range of peak presure amplitudes to sweep (MPa)
        :param title: figures title
        :return: figure handles
        '''
        # Initialize metric data container
        mdata = []

        # Create figure to plot E/I imbalance profiles
        EIfig, EIaxes = plt.subplots(1, len(arange), figsize=(2.5 * len(arange), 1.5), sharex=True, sharey=True)
        xscale = 'sqrt'

        # For each stimulus sensitivity value
        for a, EIax in zip(arange, EIaxes):
            # Set stimulus sensitivity parameter
            self.set_mech_param(a=a)
            self.plot_EI_imbalance(Pamp_range.max(), ax=EIax, legend=EIax is EIaxes[-1], xscale=xscale)
            EIax.set_title(f'a = {a:.2e}', fontsize=10)
            xscale = None

            # For each synaptic weight value
            for w in wrange:
                # Set synaptic weight parameter
                self.set_synaptic_weight(w)

                # Run comparative Pamp sweep for each stimulus distribution, compute metric
                # and append to container
                mdata.append(
                    self.run_comparative_sweep(Pamp_range, metric, **kwargs))

        # Concatenate results
        mdata = self.concatenate_outputs(
            list(itertools.product(arange, wrange * 1e6)), 
            mdata, 
            ['a', 'w (uS/cm2)']
        )

        # Plot pressure dependencies
        self.log('plotting pressure dependencies...')
        fg = sns.FacetGrid(
            mdata.reset_index(),
            row='w (uS/cm2)',
            col='a',
            aspect=1.3,
            height=2,
            sharex=False,
            sharey=False,
        )
        fg.set_titles('a={col_name:.2e}, w={row_name:.0f}uS/cm2')
        for iw, w in enumerate(wrange):
            for ix, a in enumerate(arange):
                self.plot_sweep_results(
                    mdata.loc[a, w * 1e6],
                    ax=fg.axes[iw, ix], 
                    legend=iw == wrange.size // 2 and ix == arange.size - 1,
                    xscale='sqrt',
                    marker=None,
                )
        sweepfig = fg.figure
        sweepfig.tight_layout()
        for ax in sweepfig.axes:
            ax.set_title(ax.get_title(), fontsize=9)

        # Add title
        if title is not None:
            EIfig.suptitle(title, y=1.5, fontsize=12)
            sweepfig.suptitle(title, y=1.02, fontsize=12)
        
        # Return figure handles
        return EIfig, sweepfig
    
    def find_threshold(self, s, value=1, agg='max', side='sub'):
        ''' 
        Find threshold pressure amplitude value where metric reaches a critical level
        
        :param s: metric pandas Series
        :param value: critical metric level
        :param agg: method used to aggregate matric across ndoes prior to threshold computation
        :param side: string indicating whether to return the just sub-threshold value ("sub"),
            just supra-threshold value ("supra"), or a mid-point between the two ("mid")
        :return: threshold peak pressure amplitude value (MPa)
        '''
        # If "mid" side requested, recursively call function for both "sub" and "supra" sides
        if side == 'mid':
            tsub = self.find_threshold(s, value=value, agg=agg, side='sub')
            tsupra = self.find_threshold(s, value=value, agg=agg, side='supra')
            return (tsub + tsupra) / 2
        
        # If input has multiple reps, aggregate across reps
        if self.REP_KEY in s.index.names:
            s = s.groupby(level=[k for k in s.index.names if k != self.REP_KEY]).mean()

        # Compute cross-node aggregate metric
        sagg = s.unstack(level='node').agg(agg, axis=1)

        # Select only sub-threshold or supra-threshold values, depending on "side" argument
        if side == 'sub':
            sagg = sagg[sagg < value]
        elif side == 'supra':
            sagg = sagg[sagg >= value]
        else:
            raise ValueError(f'Invalid value for "side" argument: {side}')
        
        # Group data across all non-amplitude dimensions, and extract 
        # max or min Pamp value for each group
        Pamps = sagg.reset_index(level=self.P_MPA_KEY)[self.P_MPA_KEY]
        gby = [k for k in sagg.index.names if k != self.P_MPA_KEY]
        groups = Pamps.groupby(gby)
        threshs = groups.max() if side == 'sub' else groups.min()

        # Return
        return threshs
