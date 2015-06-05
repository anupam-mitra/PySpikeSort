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
import scipy.signal

class Recording:
    """
    Represents a recording, consisting of channel(s) of data.
    
    Parameters
    ----------
    data: ndarray
        The recording signal(s)
        
    fs_Hz: float
        The sampling frequency in Hz used for recording the signal(s)
    """
    def __init__ (self, data, fs_Hz, t_spikes=None, spike_class=None):
        self.data = data
        self.t_spikes = t_spikes
        self.spike_class = spike_class
        self.fs_Hz = fs_Hz
        self.is_filtered = False
        
    def frequency_band_filter (self, flow=300, fhigh=3000, order=3):
        """
        Band pass filters a signal in the frequency domain.
        
        Parameters
        ----------
        recording :
        
        sampling_interval :

        flow :

        fhigh :
        
        order :
        """
        self.data_raw = self.data
        b, a = scipy.signal.butter(order, \
            ((flow / (self.fs_Hz / 2.0)), 
            (fhigh / (self.fs_Hz / 2.0))), 'pass')
        self.data = scipy.signal.filtfilt(b, a, self.data)
        self.is_filtered = True

