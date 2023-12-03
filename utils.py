# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-08-17 16:10:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-12-03 15:30:58

''' General utility functions. '''

import numpy as np
import pandas as pd


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