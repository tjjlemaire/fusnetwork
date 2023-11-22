# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-11-10 17:02:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-11-21 15:03:15

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Reproduction of experimental data from:
Owen et al., "Thermal constraints on in vivo optogenetic manipulations", 
Nat. Neurosci., 2019
'''

# Names of variables
Vhold_key = 'Vhold (mV)'
T_key = 'T (°C)'
IKt_key = 'IKt (pA)'
GKt_key = 'GKt (nS)'
IperT_key = 'IKt/T (pA/°C)'
GperT_key = 'GKt/T (nS/°C)'


def split_key(key):
    ''' Split variable key into variable name and unit. '''
    return re.sub('[()]', '', key).split(' ')


# Create figure
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

#%% -------------------------- Voltage dependence of iKt (Fig. 2o) --------------------------

# Extracted data
vdep_data = pd.DataFrame({
    Vhold_key: [-140, -125, -110, -95, -80, -65, -50],   # Holding potentials (mV)
    IKt_key: [-52, -37, -18, -2, 9, 14, 20],   # Light-induced current (pA)
})

# Compute conductance for each holding potential, using formula: 
# g = I / (V - EK) [pA/mV = nS]
EKt = -93  # IKt reversal potential (mV)
vdep_data[GKt_key] = vdep_data[IKt_key] / (vdep_data[Vhold_key] - EKt)  # nS

# Plot light-induced current vs. holding potential
ax = axes[0]
color = 'C0'
ax.set_title('Fig 2o: voltage-dependence of iKt', fontsize='large')
sns.lineplot(
    data=vdep_data,
    x=Vhold_key, 
    y=IKt_key, 
    marker='o',
    ax=ax,
    color=color,
)
ax.axhline(0, color='k', ls='--')
ax.set_ylabel(IKt_key, color=color)
ax.tick_params(axis='y', labelcolor=color)

# Plot conductance vs. holding potential
ax = ax.twinx()
color = 'C1'
sns.lineplot(
    data=vdep_data,
    x=Vhold_key, 
    y=GKt_key, 
    marker='o',
    ax=ax,
    color=color,
)
ax.set_ylabel(GKt_key, color=color)
ax.tick_params(axis='y', labelcolor=color)


#%% -------------------------- Temperature dependence of iKt (Fig. 2k) --------------------------

# Extracted data
Tdep_data = pd.DataFrame({
    T_key: [-3, 1, 2],   # Temperature change (°C)
    IKt_key: [-75, 25, 50],   # Temperature induced current current (pA)
})
Vhold = -50  # Voltage clamp holding potential for experiments (mV)

# Plot light-induced current vs. temperature change
ax = axes[1]
sns.despine(ax=ax)
ax.set_title('Fig 2k: temperature-dependence of iKt', fontsize='large')
sns.lineplot(
    ax=ax,
    data=Tdep_data,
    x=T_key,
    y=IKt_key,
    marker='o',
)
ax.axhline(0, color='k', ls='--')
ax.axvline(0, color='k', ls='--')

# Compute induced current per degree Celsius
Tdep_data[IperT_key] = Tdep_data[IKt_key] / Tdep_data[T_key]  # pA/°C

# Compute associated conductance per degree Celsius
Tdep_data[GperT_key] = Tdep_data[IperT_key] / (Vhold - EKt)  # nS/°C

# Annotate plot with mean values
yrel = .9
for k in [IperT_key, GperT_key]:
    var, unit = split_key(k)
    ax.text(
        0.02, yrel, 
        f'{var} = {Tdep_data[k].mean():.2f} {unit}', 
        transform=ax.transAxes)
    yrel -= .1

# Adjust figure layout and render
fig.tight_layout()
plt.show()