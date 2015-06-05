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

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

def kmeans(spike_features, num_classes, feature_scaling=True, n_jobs=-1):
    """
    Performs K means clustering using the methods of 
    sklearn.cluster.KMeans

    Parameters
    ----------
    spike_features : 
        Features extracted from the spikes.
        
    num_classes :
        Number of classes of spike present, which will be used as the
        number of clusters parameter for the clustering step.
        
    feature_scaling : boolean
        Whether to perform feature scaling
        
    n_jobs :
        Number of processes to use, default use as many as available CPUs
        using the parallelism provided by sklearn
        
    Returns
    -------
    spike_class_est:
        Estimated spike classes based on K means clustering
    
    """
    
  
    if feature_scaling:
        features = scale(spike_features, axis=0)
    else:
        features = spike_features
    
    clustering = KMeans(n_clusters=num_classes, n_init=10, n_jobs=-1)

    spike_class_est = clustering.fit_predict(features)
    return spike_class_est
