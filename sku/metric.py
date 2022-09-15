import numpy as np

def positive_split(labels:np.ndarray, predictions:None=None):
    '''
    Calculates the proportion of positive
    labels in all of the labels.
    
    
    Arguments
    ---------
    
    - labels: np.ndarray: 
        Labels should be from :code:`[0,1]` or
        :code:`[False, True]`.
    
    - predictions: None: 
        Ignored.
        Defaults to :code:`None`.
    
    
    Returns
    --------
    
    - out: float: 
        A proportion. This is calculated
        using the function 
        :code:`np.sum(labels)/labels.shape[0]`.
    
    
    '''
    return np.sum(labels)/labels.shape[0]
