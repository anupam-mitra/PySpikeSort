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

from spikesort.signals import Recording

def get_n_classes (filename, datadir):
     """
     Gets the number of spike classes in a file from the the CPGJNM 2012 dataset

     Parameters
     ----------
     filename : str
     The name of the file to read

     datadir : str
     The directory where the file is present
     """
     recording_raw, sampling_interval, spike_times, spike_class = read_file(filename, datadir)
     num_classes = np.max(np.unique(spike_class))
     return num_classes
     
def get_n_spikes (filename, datadir):
     """
     Gets the number of spikes in a file from the the CPGJNM 2012 dataset

     Parameters
     ----------
     filename : str
     The name of the file to read

     datadir : str
     The directory where the file is present
     """
     recording_raw, sampling_interval, spike_times, spike_class = read_file(filename, datadir)
     num_classes = spike_times.shape[0]
     return num_classes

def read_file (filename, datadir):
     """
     Reads a file from the CPGJNM 2012 dataset
     
     Parameters
     ----------
     filename : str
     The name of the file to read

     datadir : str
     The directory where the file is present
     """
     
     w = scipy.io.loadmat(os.path.join(datadir, filename))
     data = w['data'][0]
     fs_Hz = 24e3
     w = scipy.io.loadmat(os.path.join(datadir, 'ground_truth.mat'))
     sim_num = int(re.findall('[0-9]+', filename)[0])
     spike_class = w['spike_classes'][0][sim_num - 1][0]
     spike_times = w['spike_first_sample'][0][sim_num - 1][0] + 20

     recording = Recording(data, fs_Hz, spike_times, spike_class)
     setattr(recording, 'simulation_name', filename.replace('.mat', ''))
     return recording
