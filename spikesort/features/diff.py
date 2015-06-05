#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ${FILENAME}
#  
#  Copyright 2015 Anupam Mitra <anupam.mitra@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
# 

import numpy as np

def teo(x, k=1):
    """
    Teagre Energy Operator
    
    Parameters
    ----------
    x : ndarray
        The raw signal

    k_values : ndarray
        The values of k to use for estimating MTEO

    Returns
    -------
    te : ndarray
        The operated signal
    """
    num_samples = np.shape(x)[0]
    te = np.zeros(np.shape(x))
    te = x**2 - \
        np.concatenate([np.zeros(k), x[0 : num_samples - k]]) *\
        np.concatenate([x[k :], np.zeros(k)])
    return te

def mteo (x, k_values):
    """
    Multi resolution Teagre Energy Operator

    Parameters
    ----------
    x : ndarray
        The raw signal

    k_values : ndarray
        The values of k to use for estimating MTEO

    Returns
    -------
    tem : ndarray
        The operated signal
    """
    from scipy.signal import hamming
    from scipy.signal import lfilter

    teo_k_wise = np.zeros((len(k_values), len(x)))
    for i in range(len(k_values)):
        teo_k_wise[i, :] = teo(x, k_values[i])
        variance = np.var(teo_k_wise[i, :])
        window = hamming(4*k_values[i] + 1)
        teo_k_wise[i, :] = lfilter(window, 1, teo_k_wise[i, :]) / variance
    
    tem = np.max(teo_k_wise, axis=0)
    return tem

def firstsecond_differences (s, axis=-1):
    """
    First and second differences features
    delta_x = x[i] - x[i-1]
    delta_delta_x = delta_x[i] - delta_x[i-1]
    
    delta_x and delta_delta_x are concatenated.
    
    Parameters
    ----------
    s:
        Signal segments from which to compute first difference with
        lag. The shape should be (n_signals, n_samples)
        
    axis:
        Axis along which to compute differences.
        
    Returns
    -------
    features:
        Concatenation of feature extracted using first and second 
        differences.
    """
    s_first_diff = np.diff(s, axis=axis)
    s_second_diff = np.diff(s_first_diff, axis=axis)
    features = np.hstack(\
        (s_first_diff, s_second_diff)\
    )
    return features
    

def first_difference_lag (s, deltas, axis=-1):
    """
    First difference features with lag
    x[i] - x[i+delta]
    
    These are concatenated for all values of delta.
    
    Parameters
    ----------
    s:
        Signal segments from which to compute first difference with
        lag. The shape should be (n_signals, n_samples)
    
    deltas:
        Values of delta to use.
        
    axis:
        Axis along which to compute differences.
    
    Returns
    -------
    features:
        Concatenation of feature extracted using first differences with
        lags.
    """
    
    s = np.asanyarray(s)
    ndim = len(s.shape)

    features = []
    for delta in deltas:
        slice1 = [slice(None)]*ndim
        slice2 = [slice(None)]*ndim
        slice1[axis] = slice(delta, None)
        slice2[axis] = slice(None, -delta)
        slice1 = tuple(slice1)
        slice2 = tuple(slice2)
        
        difference_current = s[slice1] - s[slice2]
        features.append(difference_current)
    
    features = np.concatenate(features, axis=axis)
    return features


