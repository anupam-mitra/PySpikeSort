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

import cluster
import features
import signals

class SpikeSorting:
    """
    This class represents an instance of spike sorting from an
    extracellular recording
    
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
        
    n_features:
        Number of features to use for clustering.
        
    feature_clustering:
        Clustering algorithm to use for clustering. Currently only 'kMeans'
        is supported
    
    """
    
    def __init__ (self, recording, feature_extraction, feature_selection, \
        feature_clustering, n_features, samples_before=20, samples_after=44):
        self.recording = recording
        self.feature_extraction = feature_extraction
        self.feature_selection = feature_selection
        self.feature_clustering = feature_clustering
        self.n_features = n_features
        self.samples_before = samples_before
        self.samples_after = samples_after
        
    def spike_sorting (self):
        if not hasattr(self.recording, "t_spikes"):
            # TODO Need to detect spikes
            pass
        
        self.spike_features = \
            features.SpikeFeatures(\
                self.recording, self.feature_extraction, self.feature_selection)
        
        self.spike_features.extract_features()
        self.spike_features.select_features()
        
        self.clustering = \
            cluster.SpikeFeatureClustering(\
                self.recording, self.spike_features, self.n_features, 'kMeans')
        
        self.clustering.cluster_spike_features()

__all__ = ["signals", "cluster", "features", "SpikeSorting"]
