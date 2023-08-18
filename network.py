# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-13 13:37:40
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-08-17 18:11:52

import itertools
from tqdm import tqdm
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from logger import logger


class NeuralNetwork:
    ''' Interface class to a network of neurons. '''

    UM_TO_CM = 1e-4
    NA_TO_MA = 1e-6
    Acell = 11.84e3  # Cell membrane area (um2)
    mechname = 'RS'
    vrest = -71.9  # mV

    def __init__(self, nnodes, connect=True, params=None, synweight=None, verbose=True):
        '''
        Initialize a neural network with a given number of nodes.
        
        :param nnodes: number of nodes
        :param connect: whether to connect nodes (default: True)
        :param params: dictionary of model parameters (default: None)
        :param synweight: synaptic weight (uS) (default: None)
        :param verbose: verbosity flag (default: True)
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
        if params is not None:
            self.set_mech_params(params)
        self.log('initialized')
    
    def log(self, msg):
        ''' Log message with verbose-dependent logging level. '''
        logfunc = logger.info if self.verbose else logger.debug
        logfunc(f'{self}: {msg}')
    
    def __repr__(self):
        ''' String representation. '''
        return f'{self.__class__.__name__}({self.size})'
    
    def create_nodes(self, nnodes):
        ''' Create a given number of nodes. '''
        self.nodes = [h.Section(name=f'node{i}') for i in range(nnodes)]
        self.log(f'created {self.size} node{"s" if self.size > 1 else ""}')
    
    @property
    def size(self):
        return len(self.nodes)
     
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
    
    @property
    def refarea(self):
        return self.nodes[0](0.5).area()  # um2
    
    @property
    def I2i(self):
        return self.NA_TO_MA / (self.refarea * (self.UM_TO_CM**2))
    
    @property
    def i2I(self):
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
    
    def set_mech_param(self, key, value, inode=None):
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
        self.log(f'setting {key} = {value} on {nodestr}')
        for i in inode:
            setattr(self.nodes[i], f'{key}_{self.mechname}', value)
    
    def set_mech_params(self, params, inode=None):
        ''' Set multiple mechanism parameters on a specific set of nodes. '''
        for k, v in params.items():
            self.set_mech_param(k, v, inode=inode)
    
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
        ''' Record time '''
        self.probes[key] = h.Vector()
        self.probes[key].record(h._ref_t)
    
    def record_on_all_nodes(self, varname):
        '''
        Record a given variable(s) on all nodes.

        :param varname: variable name(s)
        '''
        if isinstance(varname, (tuple, list)):
            for v in varname:
                self.record_on_all_nodes(v)
            return
        self.probes[varname] = self.get_vector_list()
        for inode in range(self.size):
            self.probes[varname][inode].record(
                self.get_var_ref(inode, varname))
    
    def set_recording_probes(self):
        ''' Initialize recording probes for all nodes. '''
        # Initialize probes dictionary
        self.probes = {}

        # Assign time probe
        self.record_time()

        # Assign voltage and temperature and other probes
        self.record_on_all_nodes(['v', 'T', 'gLeak'])
    
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
            [0, 0], 
            [start, 0],
            [start, amp],
            [start + dur, amp],
            [start + dur, 0],
            [start + dur + 10, 0],
        ])
    
    def vecstr(self, values, suffix=None, detailed=True):
        ''' Return formatted string representation of node-specific values '''
        if not isinstance(values, (tuple, list, np.ndarray)):
            values = [values]
        precision = 1 if isinstance(values[0], float) else 0
        l = [f'{x:.{precision}f}' for x in values]
        if detailed:
            l = [f'    - node {i}: {x}' for i, x in enumerate(l)]
            if suffix is not None:
                l = [f'{item} {suffix}' for item in l]
            return '\n'.join(l)
        else:
            s = ', '.join(l)
            if len(l) > 1:
                s = f'[{s}]'
            if suffix is not None:
                s = f'{s} {suffix}'
            return s

    def set_stim(self, start, dur, amps):
        ''' Set stimulus per node node with specific waveform parameters. '''
        # Check that stimulus start and duration are valid
        if start < 0:
            raise ValueError('Stimulus start time must be positive')
        if dur <= 0:
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
        self.log(f'setting {dur:.2f} ms stimulus with node-specific amplitudes:\n{amps_str}')
        
        # Set stimulus vectors
        self.h_yvecs = []
        for amp in amps:
            tvec, yvec = self.get_stim_waveform(start, dur, amp).T
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
        
    def simulate(self, tstop):
        '''
        Run a simulation for a given duration.
        
        :param tstop: simulation duration (ms)
        :return: (time vector, voltage-per-node array) tuple
        '''
        # Check that simulation duration outlasts stimulus waveform
        if self.is_stim_set() and tstop < self.h_tvec.x[-1]:
            raise ValueError('Simulation duration must be longer than stimulus waveform offset')
        
        # Initialize recording probes
        self.set_recording_probes()

        # Run simulation
        self.log(f'simulating for {tstop:.2f} ms')
        h.finitialize(self.vrest)
        while h.t < tstop:
            h.fadvance()
        
        # Convert NEURON vectors to numpy arrays
        self.log('extracting output results')
        t, outvecs = self.extract_from_recording_probes()

        # Compute and log max temperature increase per node
        dT = np.max(outvecs['T'], axis=1) - np.min(outvecs['T'], axis=1)
        self.log(f'max temperature increase:\n{self.vecstr(dT, suffix="°C")}')

        # Compute and log max relative leak conductance increase per node
        gLeak_max = np.max(outvecs['gLeak'], axis=1)
        gLeak_base = np.min(outvecs['gLeak'], axis=1)
        dgLeak_rel = (gLeak_max - gLeak_base) / gLeak_base
        self.log(f'max relative leak conductance increase:\n{self.vecstr(dgLeak_rel * 100, suffix="%")}')
        
        # Count number of elicited spikes per node
        nspikes = self.extract_ap_counts(t, outvecs['v'])
        self.log(f'number of elicited spikes:\n{self.vecstr(nspikes)}')
        
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
        nspikes = []
        for v in vpernode:
            aptimes = self.extract_ap_times(t, v)
            nspikes.append(len(aptimes))
        return np.array(nspikes)

    def plot(self, t, outvecs, tref='onset', addstimspan=True, title=None):
        '''
        Plot results.
        
        :param t: time vector (ms)
        :param outvecs: dictionary of 2D arrays storing the time course of output variables across nodes
        :param tref: time reference for x-axis (default: 'onset')
        :param addstimspan: whether to add a stimulus span on all axes (default: True)
        :param title: optional figure title (default: None)
        :return: figure handle 
        '''
        self.log('plotting results')

        # Create figure
        hperax = 1.5
        naxes = 4 + int(self.is_stim_set())
        fig, axes = plt.subplots(naxes, 1, figsize=(5, hperax * naxes), sharex=True)
        sns.despine(fig=fig)
        axes[-1].set_xlabel('time (ms)')

        # Extract stimulus time-course and amplitude per node
        if self.is_stim_set():
            tvec, stimvecs = self.get_stim_vecs()
            stimbounds = np.array([tvec[1], tvec[-2]])
        
            # Expand stimulus vectors to simulation end-point
            tvec = np.append(tvec, t[-1])
            stimvecs = np.hstack((stimvecs, np.atleast_2d(stimvecs[:, -1]).T))

        # If specified, offset time to align 0 with stim onset
        if tref == 'onset' and self.is_stim_set():
                t -= stimbounds[0]
                tvec -= stimbounds[0]
                stimbounds -= stimbounds[0]
        
        # If specified, mark stimulus span on all axes
        if addstimspan and self.is_stim_set():
            for ax in axes:
                ax.axvspan(*stimbounds, fc='silver', ec=None, alpha=.3, label='stimulus')
        
        iax = 0

        # Detemrine color of time profiles
        color = 'k' if self.size == 1 else None

        # Plot stimulus time-course per node
        if self.is_stim_set():
            ax = axes[iax]
            ax.set_ylabel('Isppa (W/cm2)')
            for inode, stim in enumerate(stimvecs):
                ax.plot(tvec, stim, label=f'node{inode}', c=color)
            ax.legend()
            iax += 1

        # Plot temperature time-course per node
        ax = axes[iax]
        ax.set_ylabel('T (°C)')
        for inode, T in enumerate(outvecs['T']):
            ax.plot(t, T, label=f'node{inode}', c=color)
        iax += 1
        
        # Plot leak conductance time-course per node
        ax = axes[iax]
        ax.set_ylabel('gLeak (uS/cm2)')
        for inode, gLeak in enumerate(outvecs['gLeak']):
            ax.plot(t, gLeak * 1e6, label=f'node{inode}', c=color)
        iax += 1

        # Plot membrane potential time-course per node
        ax = axes[iax]
        ax.set_ylabel('Vm (mV)')
        for inode, v in enumerate(outvecs['v']):
            ax.plot(t, v, label=f'node{inode}', c=color)
        yb, yt = ax.get_ylim()
        ax.set_ylim(min(yb, -80), max(yt, 50))
        iax += 1

        # Plot spike raster per node
        ax = axes[iax]
        ax.set_ylabel('node index')
        for inode, v in enumerate(outvecs['v']):
            aptimes = self.extract_ap_times(t, v)
            ax.plot(aptimes, np.full_like(aptimes, inode), '|', label=f'node{inode}', c=color)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_yticks(range(self.size))

        # Add figure title, if specified
        if title is not None:
            fig.axes[0].set_title(title)
        
        # Adjust layout
        fig.tight_layout()

        # Return figure
        return fig
    
    def get_nspikes_across_sweep(self, stimdist, Isppas, start, dur, tstop):
        ''' 
        Simulate model across a range of stimulus intensities and compute the 
        number of elicited spikes per node across the range.
        
        :param stimdist: stimulus distribution vector spcifying relative intensities at each node
        :param Isppa_range: range of stimulus intensities (W/cm2)
        :param start: stimulus start time (ms)
        :param dur: stimulus duration (ms)
        :param tstop: simulation duration (ms)
        :return: 2D array with number of spikes per node across stimulus intensities
        '''
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

        # Simulate model for each stimulus vector, and count number of spikes per node
        nspikes = np.zeros_like(Isppa_vec_range)
        for i, Isppa_vec in tqdm(enumerate(Isppa_vec_range)):
            self.set_stim(start, dur, Isppa_vec)
            t, out = self.simulate(tstop)
            nspikes[i] = self.extract_ap_counts(t, out['v'])
        
        # Restore verbosity
        self.verbose = vb
        
        # Return 2D array of spike counts per node across stimulus intensities
        return nspikes
    
    def plot_sweep_results(self, Isppa_range, nspikes, title=None, ax=None):
        '''
        Plot results of a sweep across stimulus intensities.
        
        :param Isppa_range: range of stimulus intensities (W/cm2)
        :param nspikes: 2D array with number of spikes per node across stimulus intensities
        :param title: optional figure title (default: None)
        :param ax: optional axis handle (default: None)
        :return: figure handle
        '''
        # Create figure and axis, if not provided
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Plot spike counts vs Isppa
        ax.set_xlabel('Isppa (W/cm2)')
        ax.set_ylabel('# spikes')
        if title is not None:
            ax.set_title(title)
        for i, n in enumerate(nspikes.T):
            ax.plot(Isppa_range, n, '.-', label=f'node {i}')
        ax.legend()

        # Return figure
        return fig

