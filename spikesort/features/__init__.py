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

from .raw import *
from .spectral import *
from .featureselect import *
from .decomposition import *
from .diff import *
from ..signals import Recording

class SpikeFeatures:
    """
    Represents features extracted from spike waveforms
    
    Parameters
    ----------
    recording:
        Recording from which spikes are detected and sorted.
    
    feature_extraction: str
        Technique to use for extraction of spike features. Currently the 
        following are supported
        'raw' for Raw waveform
        'hw' for Haar wavelet decomposition
        'fsd' for first and second differences
        'fdl' for first difference with lag
        
    feature_selection: str
        Technique to use for selection of spike features. Currently the
        following are supported
        'var' for maximum variance
        'lt' for maximum Lilliefors test statistic
        'pca' for principal component analysis
    
    samples_before: int
        Number of samples before the peak of a spike to include in the
        spike shape
        
    samples_after: int
        Number of samples after the peak of a spike to include in the
        spike shape
        
    """
    def __init__(self, recording, feature_extraction, feature_selection,\
                samples_before=20, samples_after=44):
        self.recording = recording
        self.samples_before = samples_before
        self.samples_after = samples_after
        self.feature_extraction = feature_extraction
        self.feature_selection = feature_selection
        self.spike_waveforms = extract_waveforms(recording.t_spikes, recording.data, samples_before, samples_after)
        
    def extract_features (self):
        """
        Feature extraction step of spike sorting
        """
        if self.feature_extraction.lower() == 'raw':
            self.features = self.spike_waveforms

        elif self.feature_extraction.lower() == 'hw':
            self.features = wavelet_decomp(self.spike_waveforms)
        
        elif self.feature_extraction.lower() == 'fsd':
            self.features = firstsecond_differences(self.spike_waveforms)
            
        elif self.feature_extraction.lower() == 'fdl':
            self.features = first_difference_lag(self.spike_waveforms, [1, 3, 7])
            
        self.n_total_features = self.features.shape[1]
        
    def select_features (self):
        """
        Feature selection by ranking features based on a criterion.
        """
        if self.feature_selection.lower() == 'pca':
            self.features_selected = \
                principalcomp(self.features, \
                    n_components=self.n_total_features)
            self.index_features_selected = range(self.n_total_features)
        elif self.feature_selection.lower() == 'var':
            self.index_features_selected = variance(self.features)
        elif self.feature_selection.lower() == 'lt':
            self.index_features_selected = kstestnormal(self.features)
            
        self.features_selected = self.features[:, self.index_features_selected]
                    
    def get_top_features (self, n_features):
        features_top = self.features_selected[:, :n_features]
        return features_top

__all__ = [\
          "SpikeFeatures",
          "kstestnormal", "variance", "selectfeatures", \
          "principalcomp", "indepcomp", \
          "wavelet_decomp", "firstsecond_difference", "first_difference_lag", \
          ]
