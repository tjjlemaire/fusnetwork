# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-08-17 16:10:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-08-17 16:10:25

''' General utility functions. '''


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
