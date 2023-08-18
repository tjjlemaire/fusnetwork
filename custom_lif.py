# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-27 09:09:41
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-07-07 07:54:23

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from PySONIC.utils import logger
from PySONIC.core import PulsedProtocol

logger.setLevel(logging.INFO)

S_TO_MS = 1e3


class Stimulus:
    ''' Main interface to stimulus object '''

    def __init__(self, Iinj, tstim, toffset, *args, tstart=0., **kwargs):
        '''
        Initialization
        
        :param Iinj: injected current (pA)
        :param tstim: stimulation duration (ms)
        :param toffset: stimulation offset (ms)
        :param tstart: stimulation start time (ms)
        :param args: positional arguments to be passed to PulsedProtocol
        :param kwargs: keyword arguments to be passed to PulsedProtocol
        '''
        self.Iinj = Iinj  # pA
        self.pp = PulsedProtocol(
            tstim / S_TO_MS,  # s
            toffset / S_TO_MS,  # s
            *args,
            tstart=tstart / S_TO_MS,  # s
            **kwargs
        )
    
    ''' Create getter and setter properties for PulsedProtocol parameters '''
    @property
    def tstim(self):
        return self.pp.tstim * S_TO_MS
    
    @tstim.setter
    def tstim(self, value):
        self.pp.tstim = value / S_TO_MS
    
    @property
    def toffset(self):
        return self.pp.toffset * S_TO_MS
    
    @toffset.setter
    def toffset(self, value):
        self.pp.toffset = value / S_TO_MS
    
    @property
    def tstart(self):
        return self.pp.tstart * S_TO_MS
    
    @tstart.setter
    def tstart(self, value):
        self.pp.tstart = value / S_TO_MS
    
    @property
    def PRF(self):
        return self.pp.PRF

    @PRF.setter
    def PRF(self, value):
        self.pp.PRF = value
    
    @property
    def DC(self):
        return self.pp.DC
    
    @DC.setter
    def DC(self, value):
        self.pp.DC = value
    
    def pdict(self):
        ''' Return a dictionary of class parameters '''
        return {'Iinj': f'Iinj={self.Iinj:.2f}pA', **self.pp.pdict()}
    
    def __repr__(self) -> str:
        ''' String representation '''
        return f'{self.__class__.__name__}({", ".join(self.pdict().values())})'
    
    @property
    def tstop(self):
        ''' Return associated stimulation stop time (ms) '''
        return self.pp.tstop * S_TO_MS
    
    def stim_profile(self):
        ''' Return stimulus profile as a tuple of time (ms) and current (pA) arrays '''
        t, x = self.pp.stimProfile()
        return t * S_TO_MS, x * self.Iinj
    
    def stim_events(self):
        ''' Return stimulus events as a list of tuples of time (ms) and current (pA) '''
        evts = self.pp.stimEvents()
        return [(t * S_TO_MS, x * self.Iinj) for t, x in evts]
    
    def interpolate_events(self, t):
        '''
        Interpolate stimulus profile at given time points
        
        :param t: time points (ms)
        :return: interpolated current (pA)
        '''   
        return self.Iinj * self.pp.interpolateEvents(t / S_TO_MS)
    
    def plot(self, ax=None, c='k', t=None):
        ''' Plot stimulus profile '''
        # Retrieve axis and figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
        else:
            fig = ax.get_figure()

        # Extract stimulus profile
        if t is None:
            t, y = self.stim_profile()
        else:
            y = self.interpolate_events(t)

        # Plot stimulus profile
        ax.set_title(self, fontsize=10)
        sns.despine(ax=ax)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Iinj (pA)')
        ax.plot(t, y, c=c)
        ax.fill_between(t, np.zeros_like(y), y, color=c, alpha=0.3)

        # Return figure
        return fig


class CustomLIFNeuron:

    ''' Leaky integrate-and-fire neuron model '''

    def __init__(self, Vth=-55., Vreset=-75., taum=10, gL=10., EL = -75., tref=2, Vinit=-75.):
        ''' 
        Initialization

        :param Vth: spike threshold (mV)
        :param Vreset: reset potential (mV)
        :param taum: membrane time constant (ms)
        :param gL: leak conductance (nS)
        :param EL: leak reversal potential (mV)
        :param Vinit: initial potential (mV)
        :param tref: refractory time (ms)
        '''
        # Assign class attributes
        self.Vth = Vth
        self.Vreset = Vreset
        self.taum = taum
        self.gL = gL
        self.Vinit = Vinit
        self.EL = EL
        self.tref = tref

    def dvdt(self, v, Iinj):
        '''
        Compute voltage derivative, given a voltage value and current value
        
        :param v: membrane voltage (mV)
        :param Iinj: input current (pA)
        :return: voltage derivative (mV/ms)
        '''
        return (-(v - self.EL) + Iinj / self.gL) / self.taum

    def simulate(self, stim, tstop=None, dt=.1):
        '''
        Simulate model
        
        :param stim: stimulus object
        :param tstop (optional): total simulation duration (ms)
        :param dt (optional): simulation time step (ms)
        :return: time and membrane potential vectors, and recorded spikes times
        '''
        # If no tstop is provided, use stimulus tstop
        if tstop is None:
            tstop = stim.tstop
        
        # Generate time vector
        tvec = np.arange(0, tstop + dt, dt)  # ms

        # Interpolate stimulus over time vector
        Iinj = stim.interpolate_events(tvec)
        
        # Initialize voltage vector
        v = np.zeros(tvec.size)  # mV
        v[0] = self.Vinit

        # Initialize spikes recorder 
        rec_spikes = []

        # Initialize refractory period counter
        tr = 0.

        # For each time instant
        for i, t in enumerate(tvec[:-1]):
            # If in refractory period
            if tr > 0:
                # Reset voltage, and decrement refractory period counter
                v[i] = self.Vreset
                tr = tr - 1

            # If voltage over threshold
            elif v[i] >= self.Vth:
                # Record spike event, reset voltage, and set refractory counter
                rec_spikes.append(t)
                v[i] = self.Vreset
                tr = self.tref / dt

            # Integrate and update the membrane potential
            v[i + 1] = v[i] + self.dvdt(v[i], Iinj[i]) * dt

        # Get spike times in ms
        rec_spikes = np.array(rec_spikes)

        # Return time and membrane potential vectors, as well as recorded spikes times
        return tvec, v, rec_spikes 

    def plot_volt_trace(self, t, v, sp, ax=None):
        '''
        Plot trajetory of membrane potential for a single neuron

        :patam t: time vector (ms)
        :param v: voltage trajetory (mV)
        :param sp: spike train (ms)
        :param ax: axis handle (optional)
        :return: figure handle
        '''
        # If any spikes were recorded, increase Vm at spike location for nicer rendering
        if sp.size:
            dt = t[1] - t[0]
            ispikes = (sp / dt).astype(int) - 1
            v[ispikes] += 20

        # Grab axis if not provided
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        # Plot
        sns.despine(ax=ax)
        ax.plot(t, v, 'b', label='Vm(t)')
        ax.axhline(self.Vth, 0, 1, color='k', ls='--', label='Vth')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('voltage (mV)')
        ax.legend()
        ax.set_ylim([-80, -40])

        # Return 
        return fig


def plot_all(lif, stim, t, v, sp, figsize=(10, 4)): 
    '''
    Plot stimulus and neuron response
    
    :param lif: LIF neuron object
    :param stim: stimulus object
    :param t: time vector (ms)
    :param v: voltage trajetory (mV)
    :param sp: spike train (ms)
    :return: figure handle
    '''
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
    stim.plot(t=t, ax=axes[0])
    axes[0].set_xlabel(None)
    lif.plot_volt_trace(t, v, sp, ax=axes[1])
    return fig


if __name__ == '__main__':

    # Input current (pA)
    Iinj = 250.

    # Pulsing parameters
    tstim = 200  # stimulation time (ms)
    toffset = 200  # post-stimulation time (ms)
    tstart = 100  # stimulation start time (ms)
    PRF = 20.  # pulse repetition frequency (Hz)
    DC = .5  # pulse duty cycle (-)

    # Create stimulus object
    s = Stimulus(Iinj, tstim, toffset, PRF, DC, tstart=tstart)

    # Create LIF neuron model
    lif = CustomLIFNeuron()

    # Simulate LIF neuron model
    res = lif.simulate(s)

    # Plot results
    fig = plot_all(s, *res)
    plt.show()
