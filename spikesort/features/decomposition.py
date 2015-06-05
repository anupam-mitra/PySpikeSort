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
import sklearn.decomposition
import sklearn.preprocessing
    
def principalcomp (s, n_components=None, scale=False):
    """
    Extracts features based on decomposition in terms of principal 
    components
    
    Parameters
    ----------
    s:
        Signal segments from which to compute first difference with
        lag. The shape should be (n_signals, n_samples)
    
    n_components:
        Number of principal components to use
        
    scale:
    
    """
    if n_components == None:
        n_components =  np.shape(s)[1]
    if scale:
        s = sklearn.preprocessing.scale(s)
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(s)
    s_pca = pca.transform(s)
    return s_pca
