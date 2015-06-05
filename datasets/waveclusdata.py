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
import os
import re
import scipy.io

def parse_filename (filename):
    """
    Parses a filename and returns the name of the example and the noise level
    
    Parameters
    ----------
    filename : str
        Name of the file containing the example
        
    Returns
    -------
    example_name : str
        Name of the example
        
    example_noise : float
        Noise level in the example
        
    Example
    -------
    
    """
    matched_pattern = re.search('C_(.*)_noise([0-9]+)', filename)
    example_name = matched_pattern.group(1)
    example_noise = float('0.' + matched_pattern.group(2)) * 10
    
    return example_name, example_noise

def read_file (filename, datadir):
    """
    Reads a file from the wave_clus 2012 dataset
     
    Parameters
    ----------
    filename : str
    The name of the file to read

    datadir : str
    The directory where the file is present
    """
    w = scipy.io.loadmat(os.path.join(datadir, filename))
    data = w['data'][0]
    sampling_interval = w['samplingInterval'][0][0] 
    spike_times = w['spike_times'][0, 0][0] + 21
    spike_class = w['spike_class'][0, 0][0]
    
    fs_Hz = 1000/sampling_interval
    example_name, example_noise = parse_filename(filename)
    
    recording = Recording(data, fs_Hz, spike_times, spike_class)
    setattr(recording, 'example_name', example_name)
    setattr(recording, 'example_noise', example_noise)
    return recording


