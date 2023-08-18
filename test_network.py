# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-17 08:31:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-08-17 18:42:13

# %% Imports

import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork
from utils import pressure_to_intensity

# %%  Model definition

# Default model parameters
default_params = {
    'Tref': 37,  # reference temperature (celsius)
    'alphaT': .017,  # steady-state temperature increase (in celsius) per stimulus intensity unit
    'tauT_abs': 100,  # heat absorption time constant (ms)
    'tauT_diss': 100,  # heat dissipation time constant (ms)
    'Q10': 2,  #  Q10 coefficient for temperature dependence of leak conductance
    'gamma': 1e-5, # depolarizing force (mA/cm2) per stimulus intensity unit 
}

# Define stimulus parameters and simulation time
start = 10 # ms
dur = 150  # ms
tstop = 500  # ms
Pmax = 3e6  # Maximal acoustic pressure amplitude (Pa)

# Convert pressure to intensity
Imax = pressure_to_intensity(Pmax) / 1e4  # (W/cm2)

# %% Toy example

# Define model parameters
params = default_params.copy()

# Initialize model
model = NeuralNetwork(1, params=params)

# Define intensity distribution per node 
Isppa = Imax  # W/cm2

# Set and assign stimulus
model.set_stim(start, dur, Isppa)

# Run simulation
t, outvecs = model.simulate(tstop)

# Plot results
fig = model.plot(
    t, outvecs, 
    title=f'{model}, Isppa = {model.vecstr(Isppa, detailed=False, suffix="W/cm2")}')


# %% Adjust Q10 coefficient to achieve inhibitory effect

for Q10 in [200, 2000]:
    params['Q10'] = Q10
    model.set_mech_params(params)
    t, outvecs = model.simulate(tstop)
    fig = model.plot(t, outvecs, title=f'{model}, Q10 = {Q10}')


# %% Expand to 3 nodes

model = NeuralNetwork(3, params=params)
Isppa = np.ones(model.size) * Imax  # W/cm2
model.set_stim(start, dur, Isppa)
t, outvecs = model.simulate(tstop)
fig = model.plot(
    t, outvecs, 
    title=f'{model}, Isppa = {model.vecstr(Isppa, detailed=False, suffix="W/cm2")}')


# %% Increase syaptic weight to achieve network entrainment

for w in [.001, .005, .01]:
    model.set_synaptic_weight(w)
    t, outvecs = model.simulate(tstop)
    fig = model.plot(t, outvecs, title=f'w = {w} uS')
model.set_synaptic_weight(.002)


# %% Compare single-focus vs multi-focus stimulation (intensity-matched)

# Define stimulus distributions per node
stim_dists = {
    'single-node': np.array([1, 0, 0]),
    'multi-node': np.array([1, 1, 1])
}

# Run simulation for each stimulus distribution
for k, stim_dist in stim_dists.items():
    model.set_stim(start, dur, stim_dist * Imax)
    t, outvecs = model.simulate(tstop)
    fig = model.plot(
        t, outvecs, 
        title=f'{model}, {k} stimulation (Isppa-matched)')


# %% Extend comparison with sweep across Isppa

# Sweep parameters
Isppa_range = np.linspace(0, Imax, 15)

# Run sweep for each stimulus distribution, and store spike counts per node
nspikes = {
    k: model.get_nspikes_across_sweep(stim_dist, Isppa_range, start, dur, tstop) 
    for k, stim_dist in stim_dists.items()
}

# Plot spike counts vs Isppa for each node and each stimulus distribution
fig, axes = plt.subplots(1, len(stim_dists), sharey=True, figsize=(5 * len(stim_dists), 4))
for ax, k in zip(axes, nspikes):
    model.plot_sweep_results(Isppa_range, nspikes[k], title=f'{k} stimulation', ax=ax)


# %% Compare single-focus vs multi-focus stimulation in power-matched sweeps

# Normalize stimulus distributions to sum up to 1
stim_dists = {k: v / v.sum() for k, v in stim_dists.items()}

# Run sweep for each stimulus distribution, and store spike counts per node
nspikes = {
    k: model.get_nspikes_across_sweep(stim_dist, Isppa_range, start, dur, tstop) 
    for k, stim_dist in stim_dists.items()
}

# Plot spike counts vs Isppa for each node and each stimulus distribution
fig, axes = plt.subplots(1, len(stim_dists), sharey=True, figsize=(5 * len(stim_dists), 4))
for ax, k in zip(axes, nspikes):
    model.plot_sweep_results(Isppa_range, nspikes[k], title=f'{k} stimulation', ax=ax)

# %% Render
plt.show()
