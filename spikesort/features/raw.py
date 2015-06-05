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

def extract_waveforms(t_spikes, data, samples_before, samples_after):
    """
    Extracts spike waveforms from the signal
    
    Parameters
    ----------

    data : ndarray
        The signal from which to detect spikes
        
    t_spikes : ndarray
        The times of spikes in the signal
    
    samples_before : int
        Number of samples before each spike time to include in the waveform
        
    samples_after : int
        Number of samples after each spike time to include in the waveform
        
    Returns
    -------
    
    waveforms : ndarray
        The waveforms of the spikes extracted from the signal
    
    """
    n_spikes = t_spikes.shape[0]
    n_samples = samples_before + samples_after
    waveforms = np.empty((n_spikes, n_samples))
    for j in range(n_spikes):
        waveforms[j, :] = \
            data[t_spikes[j] - samples_before: t_spikes[j] + samples_after]
    return waveforms
