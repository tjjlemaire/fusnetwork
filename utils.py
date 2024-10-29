# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-08-17 16:10:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-10-20 15:00:50

''' General utility functions. '''

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import datetime
import matplotlib.backends.backend_pdf
from tqdm import tqdm

from logger import logger


def pressure_to_intensity(p, rho=1046.0, c=1546.3):
    '''
    Return the spatial peak, pulse average acoustic intensity (ISPPA)
    associated with the specified pressure amplitude.
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param p: pressure amplitude (Pa)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: spatial peak, pulse average acoustic intensity (W/m2)
    '''
    return p**2 / (2 * rho * c)


def intensity_to_pressure(I, rho=1046.0, c=1546.3):
    '''
    Return the pressure amplitude (in Pa) associated with the specified
    spatial peak, pulse average acoustic intensity (ISPPA).
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param I: Isppa (W/m2)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: pressure amplitude (Pa)
    '''
    return np.sqrt(I * 2 * rho * c)


def is_iterable(x):
    ''' Check if an object is iterbale (i.e. a list, tuple or numpy array) '''
    for t in [list, tuple, np.ndarray, pd.Series]:
        if isinstance(x, t):
            return True
    return False


def as_iterable(x):
    ''' Return an iterable of an object if it is not already iterable '''
    return x if is_iterable(x) else [x]


def sigmoid(x, x0, dx):
    ''' Sigmoid function '''
    return 1 / (1 + np.exp(-(x - x0) / dx))


def sigmoid_root(y, x0, dx):
    ''' Inverse sigmoid function '''
    return x0 + dx * np.log(y / (1 - y))


def exp_cdf(x, dx):
    ''' Exponential cumulative distribution function '''
    return 1 - np.exp(-x / dx)


def is_within(x, bounds):
    ''' Check if a value is within a given range '''
    return np.logical_and(x >= bounds[0], x <= bounds[1])


def signed_sqrt(x):
    ''' Signed square root function. '''
    if is_iterable(x):
        return np.array([signed_sqrt(xi) for xi in x])
    return np.sqrt(x) if x >= 0 else -np.sqrt(-x)


def sqrtspace(start, stop, n):
    ''' Generate range of values linearly spaced in sqrt space '''
    return np.linspace(np.sqrt(start), np.sqrt(stop), n)**2


def restrict_trace(x, y, ymin, ymax, full_output=False):
    '''
    Restrict a time course to a given range of values by 
    (1) setting values outside the range to NaN, and
    (2) adding interpolated points at the boundaries.

    :param x: time vector
    :param y: value vector
    :param ymin: minimal value
    :param ymax: maximal value
    :param full_output: whether to return also return a dictionary of booleans
     indicating whether trace is clipped on each y boundary
    :return: restricted time and value vectors (and optional dictionary of booleans)
    '''
    # Set up boundaries dictionary
    ylims = {'lb': ymin, 'ub': ymax}

    # Create boolean masks for values above and below the bounds
    masks = {
        'lb': y < ylims['lb'],
        'ub': y > ylims['ub']
    }

    # Determine if any values fall on either side of the bounds
    is_clipped = {k: mask.any() for k, mask in masks.items()}

    # If values fall outside the bounds
    if any(is_clipped.values()):

        # Find the indices where the trajectory crosses the bounds
        transitions = []
        for kind, mask in masks.items():
            mask_diff = np.diff(mask.astype(int))
            idxs = np.where(mask_diff != 0)[0]
            transitions.append(
                pd.Series(index=idxs, data=kind, name='kind'))
        transitions = pd.concat(transitions, axis=0).sort_index()

        # Copy x and y arrays into points dataframe, and set 
        # out-of-bounds y values to NaN
        points = pd.DataFrame({'x': x, 'y': y})
        for kind, mask in masks.items():
            points['y'][mask] = np.nan

        # Add interpolated points at the locations of boundary crossings
        for i, kind in transitions.items():
            yb = ylims[kind]
            finterp = interp1d(y[i:i + 2], x[i:i + 2])
            points.loc[len(points)] = (finterp(yb), yb)

        # Sort points by x value
        points = points.sort_values(by='x')

        # Extract new x and y arrays
        x, y = points['x'].values, points['y'].values

    # Return
    if full_output:
        return x, y, is_clipped
    else:
        return x, y


def expand_range(x, factor=0.1):
    '''
    Expand range by relative factor around its center point

    :param x: range bounds vector
    :param factor: relative expansion factor (default: 0.1)
    :return: expanded range bounds vector
    '''
    # Unpack bounds
    lb, ub = x
    # Compute center and halfwidth
    center = (lb + ub) / 2
    halfwidth = (ub - lb) / 2
    # Expand halfwidth
    halfwidth *= (1 + factor)
    # Return expanded bounds
    return np.array([center - halfwidth, center + halfwidth])


def save_figs_book(figsroot, figs, suffix=None):
    ''' Save figures dictionary as consecutive pages in single PDF document. '''
    now = datetime.datetime.now()
    today = now.strftime('%Y.%m.%d')
    figsdir = os.path.join(figsroot, today)
    if not os.path.isdir(figsdir):
        os.mkdir(figsdir)
    fcode = 'figs'
    if suffix is not None:
        fcode = f'{fcode}_{suffix}'
    fname = f'{fcode}.pdf'
    fpath = os.path.join(figsdir, fname)
    file = matplotlib.backends.backend_pdf.PdfPages(fpath)
    logger.info(f'saving figures in {fpath}:')
    for v in tqdm(figs.values()):
        file.savefig(v, transparent=True, bbox_inches='tight')
    file.close()