# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
        
    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """    
    # Compute split indices
    Ndata = x.shape[0]
    Ntr = int(np.rint(Ndata*ratio))
    Nte = Ndata-Ntr
    np.random.seed(seed)    
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    x_tr = x[trIdx]
    y_tr = y[trIdx]
    x_te = x[teIdx]
    y_te = y[teIdx]
    return x_tr,x_te,y_tr,y_te
