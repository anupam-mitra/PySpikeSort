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
import heapq

def kstestnormal (spike_features):
    """
    Ranks features using a Lilliefors test for
    normality
    
    Parameters
    ----------
    spike_features : ndarray
        The spike features
        
    Returns
    -------
    index_features_sorted : ndarray
        The indices of selected features in order of decreasing KS test
        statistic
    """
    num_total_features =  np.shape(spike_features)[1]
    from statsmodels.stats.diagnostic import kstest_normal
    ksD = np.zeros(num_total_features)
    ksp = np.zeros(num_total_features)
    for f in range(num_total_features):
        ksD[f], ksp[f] = kstest_normal(spike_features[:,f], pvalmethod='approx')

    index_features_sorted = heapq.nlargest(num_total_features, range(len(ksD)), ksD.take)
    return index_features_sorted
    
def variance (spike_features):
    """
    Feature features using variance
    
    Parameters
    ----------
    spike_features : ndarray
        The spike features
    
       
    Returns
    -------
    index_features_sorted : ndarray
        The indices of selected features in order of decreasing KS test
        statistic
    """
    num_total_features =  np.shape(spike_features)[1]
    spike_features_variance = np.var(spike_features, axis=0)

    index_features_sorted = heapq.nlargest(num_total_features, \
        range(len(spike_features_variance)), spike_features_variance.take)
    return index_features_sorted
    
def selectfeatures (spike_features, n_features, criterion='Var'):
    """
    Selects features which give the maximum score on a criterion
    
    Parameters
    ----------
    spike_features :
        The spike features
        
    n_features : int
        Number of features to select
    
    criterion :
        Criterion to use for selecting features. Currently 
        'kstest', 'var' are supported
        
    Returns
    -------
    index_features_selected :
        Indices of the selected spike features
    
    spike_features_selected :
        Selected spike features based on maximum score on criterion
    """
    num_total_features =  np.shape(spike_features)[1]
    
    index_features_sorted = None
    if criterion.lower() == 'var':
        index_features_sorted = variance(spike_features)
    elif criterion.lower() == 'lt':
        index_features_sorted = kstestnormal(spike_features)
        
    if index_features_sorted == None:
        print '***** Something Wrong *****'
    
    index_features_selected = index_features_sorted[:n_features]
    spike_features_selected = spike_features[:, index_features_selected]
    return index_features_selected, spike_features_selected
