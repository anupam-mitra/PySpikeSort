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
import scipy.io
import os

import datasets

def detect_spikes (data, threshold=None):
    """
    Detects spikes.
    This is not completely implemented.
    
    Parameters
    ----------
    data : ndarray
        The signal from which to detect spikes

    threshold : float
        The threshold to use for detecting spikes
        
    Returns
    -------
    t_spikes_detect : ndarray
        Times of detect spikes
    
    """
    import numpy as np
    import statsmodels.robust.scale

    if threshold == None:
        # Default threshold assuming a normal distribution
        # sigmaN = np.median(np.abs(data))/0.6745
        mad = 1.4826 * statsmodels.robust.scale.mad(np.abs(data))
        threshold = 3.38 * mad

    index_cross_threshold = np.where(data > threshold)[0]

    t_spikes_detect = []
    i = 0
    while i <= len(index_cross_threshold)-1:
        j = i
        while (index_cross_threshold[j+1] - index_cross_threshold[j] == 1):
            j = j+1
            if j >= len(index_cross_threshold)-1:
                break
        t = index_cross_threshold[i] + \
            np.argmax(recording[index_cross_threshold[i]:index_cross_threshold[j]+1])
        i = j+1
        
        t_spikes_detect.append(t)
    return np.asarray(t_spikes_detect)

