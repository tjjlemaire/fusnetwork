# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-13 13:37:40
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-11-29 10:20:31

import itertools
from tqdm import tqdm
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from logger import logger


def get_NetStim(freq, start=0., number=1e9, noise=0):
    '''
    Wrapper around NetStim allowing to set parameters in 1-liner.
    
    :param freq: spiking frequency of pre-synaptic drive (Hz)
    :param start (optional): start time (ms)
    :param number (optional): total number of pre-synaptic spikes
    :return: NetStim object
    '''
    stim = h.NetStim()
    stim.number = number
    stim.start = start # ms
    stim.interval = 1e3 / freq  # ms
    stim.noise = noise
    return stim


class NeuralNetwork:
    ''' Interface class to a network of neurons. '''

    # Conversion factors
    UM_TO_CM = 1e-4
    NA_TO_MA = 1e-6
    S_TO_US = 1e6
    S_TO_MS = 1e3

    # Default model parameters
    Acell = 11.84e3  # Cell membrane area (um2)
    mechname = 'RS'  # NEURON mechanism name
    vrest = -71.9  # neurons resting potential (mV)

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
            'idrive': 'driving current (mA/cm2)',
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
    
    def create_nodes(self, nnodes):
        ''' Create a given number of nodes. '''
        self.nodes = [h.Section(name=f'node{i}') for i in range(nnodes)]
        self.log(f'created {self.size} node{"s" if self.size > 1 else ""}')
    
    @property
    def size(self):
        return len(self.nodes)

    @property
    def refarea(self):
        ''' Surface area of first node in the model (um2). '''
        return self.nodes[0](0.5).area()  # um2
    
    @property
    def I2i(self):
        ''' Current to current density conversion factor (nA to mA/cm2). '''
        return self.NA_TO_MA / (self.refarea * (self.UM_TO_CM**2))
    
    @property
    def i2I(self):
        ''' Current density to current conversion factor (mA/cm2 to nA). '''
        return 1 / self.I2i
    
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
     
    def connect(self, ipresyn, ipostsyn, weight=0.002):
        '''
        Connect a source node to a target node with a specific synapse model
        and synaptic weight.

        :param ipresyn: index of the pre-synaptic node
        :param ipostsyn: index of the post-synaptic node
        :param weight: synaptic weight (uS)
        '''
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
        nc.weight[0] = weight * self.refarea / self.Acell  # synaptic weight (uS)

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
        return hasattr(self, 'netcon_objs')
    
    def set_synaptic_weight(self, w):
        ''' Set synaptic weight on all connections. '''
        if not self.is_connected():
            raise ValueError('Network nodes are not connected')
        self.log(f'setting all synaptic weights to {w:.2f} uS')
        for nc in self.netcon_objs:
            nc.weight[0] = w * self.refarea / self.Acell
    
    def get_vector_list(self):
        ''' Return model-sized list of NEURON vectors. '''
        return [h.Vector() for node in self.nodes]
    
    def get_var_ref(self, inode, varname):
        ''' 
        Get reference to specific variable on a given node.

        :param inode: node index
        :param varname: variable name
        '''
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
        ''' List disabled membrane currents in the model (i.e. those with null
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
        
        :return: 2-tuple with:
            - time array (ms)
            - dictionary of output 2D arrays storing the time course of all recording variables across nodes
        '''
        outvecs = {}
        for k, v in self.probes.items():
            if isinstance(v, list):
                outvecs[k] = np.array([x.to_python() for x in v])
            else:
                outvecs[k] = np.array(v.to_python())
        return outvecs.pop('t'), outvecs

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
    
    def vecstr(self, values, dev=None, prefix=None, suffix=None, detailed=True):
        ''' Return formatted string representation of node-specific values '''
        # Format input as iterable if not already
        if not isinstance(values, (tuple, list, np.ndarray)):
            values = [values]
        
        # Determine logging precision based on input type
        precision = 1 if isinstance(values[0], float) else 0

        # Format values as strings
        l = [f'{x:.{precision}f}' if x is not None else 'N/A' for x in values]
        if dev is not None:
            devl = [f'±{x:.{precision}f}' if x is not None else '' for x in dev]
            l = [f'{x} {dev}' for x, dev in zip(l, devl)]

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
        
    def simulate(self, tstop=None):
        '''
        Run a simulation for a given duration.
        
        :param tstop: simulation duration (ms)
        :return: (time vector, voltage-per-node array) tuple
        '''
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
        t, outvecs = self.extract_from_recording_probes()

        # Compute and log max temperature increase per node
        dT = self.compute_metric(t, outvecs, 'dT')
        self.log(f'max temperature increase:\n{self.vecstr(dT, prefix="ΔT =", suffix="°C")}')
        
        # Count number of elicited spikes and average firing rate per node
        tspikes = self.extract_ap_times(t, outvecs['v'])  # ms
        nspikes = np.array([len(ts) for ts in tspikes])
        self.log(f'number of elicited spikes:\n{self.vecstr(nspikes, prefix="n =", suffix="spikes")}')
        FRs = []
        for ts in tspikes:
            if len(ts) > 1:
                FR = 1 / np.diff(ts) * self.S_TO_MS  # Hz
            else:
                FR = None
            FRs.append(FR)
        mu_FRs = [FR.mean() if FR is not None else None for FR in FRs]  # Hz
        sigma_FRs = [FR.std() if FR is not None else None for FR in FRs]  # Hz
        self.log(f'elicited firing rate:\n{self.vecstr(mu_FRs, dev=sigma_FRs, prefix="FR =", suffix="Hz")}')
        
        # Return time and dictionary arrays of recorded variables
        return t, outvecs
    
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
    
    def filter_output(self, outvecs, **kwargs):
        '''
        Filter simulation output variables according to inclusion and exclusion criteria.

        :param outvecs: dictionary of 2D arrays storing the time course of output variables across nodes
        :param kwargs: inclusion and exclusion criteria
        :return: filtered output dictionary
        '''
        # Extract list of keys from inclusion and exclusion criteria
        keys = self.filter_record_keys(**kwargs)

        # Get intersection between keys and output dictionary keys 
        # (in case some currents are disabled)
        keys = [k for k in keys if k in outvecs.keys()]

        # Return filtered output dictionary
        return {k: outvecs[k] for k in keys}

    def plot_results(self, t, outvecs, tref='onset', gmode='abs', addstimspan=True, title=None, **kwargs):
        '''
        Plot the time course of variables recorded during simulation.
        
        :param t: time vector (ms)
        :param outvecs: dictionary of 2D arrays storing the time course of output variables across nodes
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
        outvecs = self.filter_output(outvecs, **kwargs)

        # Log
        self.log('plotting results')

        # Create figure
        hrow = 1.5
        wcol = 3.5
        hastemps = 'T' in outvecs
        hasconds = 'conductances' in self.record_dict and any(k in outvecs for k in self.record_dict['conductances'])
        hascurrs = 'currents' in self.record_dict and any(k in outvecs for k in self.record_dict['currents'])
        hasvoltages = 'v' in outvecs
        nrows = int(self.is_stim_set()) + int(hastemps) + int(hasconds) + int(hascurrs) + int(hasvoltages)
        refkey = list(outvecs.keys())[0]
        ncols = outvecs[refkey].shape[0]
        assert ncols == self.size, f'number of nodes ({self.size}) does not match number of output voltage traces ({ncols})'
        fig, axes = plt.subplots(nrows, ncols, figsize=(wcol * ncols + 1.5, hrow * nrows), sharex=True, sharey='row')
        if ncols == 1:
            axes = np.atleast_2d(axes).T
        sns.despine(fig=fig)
        for ax in axes[-1]:
            ax.set_xlabel('time (ms)')
        for inode, ax in enumerate(axes[0]):
            ax.set_title(f'node {inode}')

        # Define legend keyword arguments
        leg_kwargs = dict(
            bbox_to_anchor=(1.0, .5),
            loc='center left',
            frameon=False,
        )

        # Extract stimulus time-course and amplitude per node
        if self.is_stim_set():
            tvec, stimvecs = self.get_stim_vecs()
            stimbounds = np.array([tvec[1], tvec[-2]])
        
            # Expand stimulus vectors to simulation end-point
            tvec = np.append(tvec, t[-1])
            stimvecs = np.hstack((stimvecs, np.atleast_2d(stimvecs[:, -1]).T))

            # Extract max stimulus intensity per node, and complete column titles
            Isppas = np.max(stimvecs, axis=1)
            for ax, Isppa in zip(axes[0], Isppas):
                ax.set_title(f'{ax.get_title()} - Isppa = {Isppa:.1f} W/cm2')

        # If specified, offset time to align 0 with stim onset
        if tref == 'onset' and self.is_stim_set():
                t -= stimbounds[0]
                tvec -= stimbounds[0]
                stimbounds -= stimbounds[0]
        
        # If specified, mark stimulus span on all axes
        if addstimspan and self.is_stim_set():
            for irow, axrow in enumerate(axes):
                for ax in axrow:
                    ax.axvspan(
                        *stimbounds, fc='silver', ec=None, alpha=.3,
                        label='stimulus' if irow == 0 else None)
        
        # Initialize axis row index
        irow = 0

        # Plot stimulus time-course per node
        if self.is_stim_set():
            axrow = axes[irow]
            axrow[0].set_ylabel('Isppa (W/cm2)')
            for ax, stim in zip(axrow, stimvecs):
                ax.plot(tvec, stim, c='k')
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Plot temperature time-course per node
        if hastemps:
            axrow = axes[irow]
            axrow[0].set_ylabel('T (°C)')
            for ax, T in zip(axrow, outvecs['T']):
                ax.plot(t, T, c='k')
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
                if condkey in outvecs:
                    for ax, g in zip(axrow, outvecs[condkey]):
                        if condkey.endswith('bar'):
                            label = f'\overline{{g_{{{condkey[1:-3]}}}}}'
                        else:
                            label = f'g_{{{condkey[1:]}}}'
                        if gmode == 'rel':
                            if g[0] == 0:
                                logger.warning(f'Cannot compute relative conductance for {label}: baseline is 0')
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                g = g / g[0] * 100
                        elif gmode == 'norm':
                            g = g / np.max(g) * 100
                        ax.plot(t, g, label=f'${label}$')
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
                if ckey in outvecs:
                    for ax, i in zip(axrow, outvecs[ckey]):
                        ax.plot(t, i, label=f'$i_{{{ckey[1:]}}}$', color=color)
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # Plot membrane potential time-course per node
        if hasvoltages:
            axrow = axes[irow]
            axrow[0].set_ylabel('Vm (mV)')
            for ax, v in zip(axrow, outvecs['v']):
                ax.plot(t, v, c='k', label='trace')
                aptimes = self.extract_ap_times(t, v)
                ax.plot(aptimes, np.full_like(aptimes, 70), '|', c='dimgray', label='spikes')
                yb, yt = ax.get_ylim()
                ax.set_ylim(min(yb, -80), max(yt, 50))
            axrow[-1].legend(**leg_kwargs)
            irow += 1

        # # Plot spike raster per node
        # ax = axes[iax]
        # ax.set_ylabel('node index')
        # for inode, v in enumerate(outvecs['v']):
        #     aptimes = self.extract_ap_times(t, v)
        #     ax.plot(aptimes, np.full_like(aptimes, inode), '|', label=f'node{inode}', c='k')
        # ax.set_ylim(-0.5, self.size - 0.5)
        # ax.set_yticks(range(self.size))

        # Add figure title, if specified
        if title is None and ncols > 1:
            title = self
        if title is not None:
            fig.suptitle(title)
        
        # Adjust layout
        fig.tight_layout()

        # Return figure
        return fig
    
    def run_sweep(self, Isppas, stimdist=None, **kwargs):
        ''' 
        Simulate model across a range of stimulus intensities and return outputs.
        
        :param Isppas: range of stimulus intensities (W/cm2)
        :param stimdist (optional): vector spcifying relative stimulus intensities at each node. 
            If not provided, all nodes will be stimulated with the same intensity.
        :param kwargs: optional arguments passed to "set_stim" and "simulate" methods
        :return: 2D array with number of spikes per node across stimulus intensities
        '''
        # If stimulus distribution vector not provided, assume uniform distribution across nodes
        if stimdist is None:
            stimdist = [1] * self.size
        
        # Check that stimulus distribution vector is valid
        if len(stimdist) != self.size:
            raise ValueError(f'Number of stimulus distribution values {len(stimdist)} does not match number of nodes {self.size}')
        if any(amp < 0 for amp in stimdist):
            raise ValueError('Stimulus distribution values must be positive')
        if sum(stimdist) == 0:
            raise ValueError('At least one stimulus value must be non-zero')
        
        self.log(f'running simulation sweep across {len(Isppas)} stimulus intensities')

        # Generate 2D array of stimulus vectors for each stimulus intensity
        Isppa_vec_range = np.dot(np.atleast_2d(Isppas).T, np.atleast_2d(stimdist))

        # Disable verbosity during sweep
        vb = self.verbose
        self.verbose = False

        # Simulate model for each stimulus vector, assemble outputs list
        tstop = kwargs.pop('tstop', None)
        outs = []
        for Isppa_vec in tqdm(Isppa_vec_range):
            self.set_stim(Isppa_vec, **kwargs)
            t, out = self.simulate(tstop=tstop)
            outs.append(out)
        
        # Restore verbosity
        self.verbose = vb

        # Return time vector and list of outputs across stimulus intensities
        return t, outs
    
    def compute_metric(self, t, out, metric):
        ''' Compute metric across nodes from a given output dictionary. '''
        if metric == 'nspikes':
            return self.extract_ap_counts(t, out['v'])
        elif metric == 'dT':
            return np.max(out['T'], axis=1) - np.min(out['T'], axis=1)
        else:
            raise ValueError(f'Invalid metric: {metric}')
    
    def plot_sweep_results(self, Isppas, t, outs, metric, title=None, ax=None, width=4, height=2):
        '''
        Plot results of a sweep across stimulus intensities.
        
        :param Isppas: range of stimulus intensities (W/cm2)
        :param t: time vector (ms)
        :param outs: list of outputs across stimulus intensities
        :param metric: metric(s) to plot
        :param title: optional figure title (default: None)
        :param ax: optional axis handle (default: None)
        :return: figure handle
        '''
        # If multiple metrics provided, recursively call function for each metric
        if isinstance(metric, (tuple, list)) and len(metric) > 1:
            # Create figure
            nmetrics = len(metric)
            fig, axes = plt.subplots(nmetrics, 1, figsize=(width, height * nmetrics), sharex=True)
            if title is not None:
                fig.suptitle(title)

            # Plot each metric on a separate axis
            for ax, m in zip(axes, metric):
                self.plot_sweep_results(Isppas, t, outs, m, ax=ax)
            
            # Return figure
            return fig
        
        # Create figure and axis, if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Determine number of nodes and stimulus intensities
        nstims = len(outs)
        if nstims != len(Isppas):
            raise ValueError(
                f'number of stimulus intensities ({nstims}) does not match number of outputs ({len(Isppas)})')
        k = list(outs[0].keys())[0]
        nnodes = len(outs[0][k])

        # Compute metric across nodes and stimulus intensities
        m = np.zeros((nstims, nnodes))
        for istim, out in enumerate(outs):
            m[istim] = self.compute_metric(t, out, metric)

        # Plot metric vs Isppa
        ax.set_xlabel('Isppa (W/cm2)')
        ylabel = metric
        if metric == 'dT':
            ylabel += ' (°C)'
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)        
        for inode, vec in enumerate(m.T):
            ax.plot(Isppas, vec, '.-', label=f'node {inode}')

        # Add legend if multiple nodes
        if nnodes > 1:
            ax.legend()

        # Return figure
        return fig

