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
import pywt

def wavelet_decomp (s, wavelet='haar', levels=4):
    """
    Wavelet decomposition based features
    
    Parameters
    ----------
    s:
        Signal segments from which to compute first difference with
        lag. The shape should be (n_signals, n_samples)
        
    wavelet: str
        Wavelet basis to use for decomposition
        
    levels: int
        Level of wavelet decomposition
    """

    n_features = np.shape(s)[1]
    n_spikes = np.shape(s)[0]
    features = np.empty((n_spikes, n_features))
    for s in range(n_spikes):
        wd = pywt.wavedec(s[s, :], wavelet=wavelet, level=levels)
        features[s,:] = np.hstack(wd)
    return features

