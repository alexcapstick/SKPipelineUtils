from __future__ import annotations
import sklearn
import numpy as np
import pandas as pd
import copy
import typing

from .preprocessing import StandardGroupScaler
from .transformer_wrapper import SKTransformerWrapperDD
from .utils import partialclass

class _DropNaNRowsDD:
    '''
    The rows in which no NaNs 
    appear across all given arguments
    will be returned.

 
    '''
    def fit(self, X:typing.Any, y:None=None):
        '''
        Ignored.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Any```: 
            Ignored
        
        - ```y```: ```None```, optional:
            Ignored. 
            Defaults to ```None```.
        
        
        '''
        return
    
    def transform(self, *args) -> typing.Union[list, np.ndarray]:
        '''
        This will remove the NaN rows
        across all arguments.
        
        Arguments
        ---------

        - ```*args```: ```np.ndarray```:
            Arguments to filter the NaN
            rows from.


        Returns
        --------
        
        - ```out```: ```typing.Union[list, np.ndarray]```:
            Arguments with same structure 
            as inputs.
        
        
        '''
        nan_rows = []
        for ia, arg in enumerate(args):
            try:
                if len(arg.shape)>1:
                    nan_rows.append(np.any(pd.isnull(arg), axis=1))
                else:
                    nan_rows.append(pd.isnull(arg))
            except Exception as e:
                raise type(e)(f'Failed on argument {ia}: {str(e)}')
        nan_rows = np.any(np.vstack(nan_rows), axis=0)
        output = [arg[~nan_rows] for arg in args]
        if len(output)==1:
            output = output[0]
        return output



DropNaNRowsDD = partialclass(SKTransformerWrapperDD, 
                                transformer=_DropNaNRowsDD, 
                                all_key_transform=True)





StandardGroupScalerDD = partialclass(SKTransformerWrapperDD, 
                                transformer=StandardGroupScaler, 
                                all_key_transform=False)










class NumpyToDict(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        '''
        This function will transform two numpy arrays
        containing the inputs and targets to a dictionary
        containing the same arrays.
        
        
        '''
        self.X = None
        self.y = None
        return
    
    def fit(self, 
                X:typing.Union[np.ndarray, None]=None, 
                y:typing.Union[np.ndarray, None]=None) -> NumpyToDict:
        '''
        This will save the numpy arrays to the class, ready
        to return them as a dictionary in the ```.transform()``` 
        method
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Union[np.ndarray, None]```, optional:
            The inputs. 
            Defaults to ```None```.
        
        - ```y```: ```typing.Union[np.ndarray, None]```, optional:
            The targets. 
            Defaults to ```None```.
        
        
        '''
        self.X = X
        self.y = y
        return self
    
    def transform(self, X:typing.Any=None) -> typing.Dict[str, np.ndarray]:
        '''
        _summary_
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Any```: 
            Ignored.
            Defaults to ```None```.
        
        
        Returns
        --------
        
        - ```out```: ```typing.Dict[str, np.ndarray]``` : 
            A dictionary of the form:
            ```
            {'X': self.X, 'y': self.y}
            ```
        
        
        '''
        return {'X': self.X, 'y': self.y}