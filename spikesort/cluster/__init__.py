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
import sklearn.metrics.cluster

from . import euclidean
from .. import features
from .. import signals


class SpikeFeatureClustering:
    """
    Represents an instance of spike feature clustering process.
    
    Parameters
    ----------
    recording:
        Recording from which spikes are detected and sorted.
        
    spike_features:
        Object containing the extracted and ranked spike features.
    
    n_features:
        Number of features to use for clustering.
        
    cluster_algo:
        Clustering algorithm to use for clustering. Currently only 'kMeans'
        is supported
    """
    def __init__(self, recording, spike_features, n_features, cluster_algo):
        self.recording = recording
        self.spike_features = spike_features
        self.n_features = n_features
        self.n_spike_classes = np.unique(self.recording.spike_class).shape[0]
        self.n_spikes = self.recording.spike_class.shape[0]
        self.cluster_algo = cluster_algo
        
    def cluster_spike_features (self):
        features = self.spike_features.get_top_features(self.n_features)
        
        if self.cluster_algo.lower() == 'kmeans':
            self.spike_class_est = euclidean.kmeans(features, self.n_spike_classes, feature_scaling=True)

        # Not implemented at present, random assignment of class labels
        else:
            self.spike_class_est = np.random.choice(np.arange(0, self.n_spike_classes), self.n_spikes)

        self.ami = sklearn.metrics.adjusted_mutual_info_score(\
                    self.recording.spike_class, self.spike_class_est)
        self.ari = sklearn.metrics.cluster.adjusted_rand_score(\
                    self.recording.spike_class, self.spike_class_est)        


__all__ = [\
            "kmeans", \
            "SpikeClustering",\
        ]
