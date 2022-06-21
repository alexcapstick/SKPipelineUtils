from __future__ import annotations
import sklearn
import numpy as np
import copy
import typing

from .preprocessing import StandardGroupScaler

class DropNaNRowsDD(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, 
                    other_keys:list=[],
                    ) -> None:
        '''
        Sklearn transformer that removes all rows
        in which one of the elements contains a 
        NaN value.
        
        
        
        Arguments
        ---------
        
        - ```other_keys```: ```list```, optional:
            The keys specified in ```'other_keys'```
            will also be filtered based on the ```X``` and ```y```
            ```NaN``` values .
            Defaults to ```[]```.
        
        
        
        
        '''
        self.other_keys = other_keys
        return
    
    def fit(self, 
            X:None=None, 
            y=None,
            ) -> DropNaNRowsDD:
        '''
        This method is ignored and only 
        serves to allow compatibility with 
        ```sklearn.pipeline.Pipeline```.
        
        
        Arguments
        ---------
        
        - ```X```: ```None```, optional:
            Ignored. 
            Defaults to ```None```.
        
        - ```y```: ```_type_```, optional:
            Ignored. 
            Defaults to ```None```.
        
        
        Returns
        --------
        
        - ```self```: ```DropNaNRowsDD``` : 
            This object.
        
        
        '''
        return self
    
    def transform(self, 
                    X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]],
                    )-> typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]:
        '''
        This will transform the data by removing all rows
        in which one of the elements contains a 
        NaN value.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]```: 
            A dictionary containing the data.
            Either:
            ```
            X = {
                    'labelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    'unlabelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    }
            ```
            Or
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.
        
        
        Returns
        --------
        
        - ```X_out```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]``` : 
            The data dictionary with the same structure as ```X```,
            with all NaN values in ```X``` data removed.
        

        '''
        X_out = copy.deepcopy(X)
        if 'labelled' in X_out:
            nan_labelled = ( np.any(np.isnan(X_out['labelled']['X']),axis=1) | np.isnan(X_out['labelled']['y']) )
            X_out['labelled']['X'] = X_out['labelled']['X'][~nan_labelled]
            X_out['labelled']['y'] = X_out['labelled']['y'][~nan_labelled]
            for key in self.other_keys:
                if key in X_out['labelled']:
                    X_out['labelled'][key] = X_out['labelled'][key][~nan_labelled]

        if 'unlabelled' in X_out:
            nan_unlabelled = np.any(np.isnan(X_out['unlabelled']['X']),axis=1)
            X_out['unlabelled']['X'] = X_out['unlabelled']['X'][~nan_unlabelled]
            if key in X_out['unlabelled']:
                X_out['unlabelled'][key] = X_out['unlabelled'][key][~nan_unlabelled]

        if 'X' in X_out:
            nan_labelled = ( np.any(np.isnan(X_out['X']),axis=1) | np.isnan(X_out['y']) )
            X_out['X'] = X_out['X'][~nan_labelled]
            X_out['y'] = X_out['y'][~nan_labelled]
            for key in self.other_keys:
                if key in X_out:
                    X_out[key] = X_out[key][~nan_labelled]

        return X_out






















class StandardGroupScalerDD(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, 
                    semi_supervised:bool=False, 
                    group_name:typing.Union[None, str]=None):
        '''
        A wrapper around StandardGroupScaler that 
        deals with semi-supervised data.
        This class fits the scaler on the unlabelled data
        and transforms to labelled and unlabelled data.
        
        
        Arguments
        ---------
        
        - ```semi_supervised```: ```bool```, optional:
            Whether the dictionary of data contains keys
            ```'labelled'``` and ```'unlabelled'```.
            For example, if ```semi_supervised=True```:
            ```
            X = {
                    'labelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    'unlabelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    }
            ```
            If ```semi_supervised=False```:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```
            Defaults to ```False```.
        
        '''
        self.scalar = StandardGroupScaler()
        self.semi_supervised = semi_supervised
        self.group_name = group_name
        return
    
    def fit(self, 
            X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]], 
            y:None=None,
            ) -> StandardGroupScalerDD:
        '''
        Fit the scaler statistics to the unlabelled data.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]```: 
            A dictionary containing the data.
            For example, if ```semi_supervised=True```:
            ```
            X = {
                    'labelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    'unlabelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    }
            ```
            If ```semi_supervised=False```:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.
        
        - ```y```: ```_type_```, optional:
            Ignored. 
            Defaults to ```None```.
        
        
        
        Returns
        --------
        
        - ```self```: ```StandardGroupScalerDD``` : 
            The fitted scaler.
        
        
        '''
        if self.semi_supervised:
            if self.group_name in X['unlabelled']:
                groups = X['unlabelled'][self.group_name]
            else:
                groups = None
            self.scalar.fit(X['unlabelled']['X'], groups=groups)
        
        else:
            if self.group_name in X:
                groups = X[self.group_name]
            else:
                groups = None
            self.scalar.fit(X['X'], groups=groups)
        return self
    
    def transform(self, 
                    X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]],
                    ) -> typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]:
        '''
        Transforms the labelled and unlabelled data based on the fitted
        scaler.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]```: 
            A dictionary containing the data.
            Either:
            ```
            X = {
                    'labelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    'unlabelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    }
            ```
            Or
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.
        
        
        
        Returns
        --------
        
        - ```out```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]``` : 
            The data dictionary with the same structure as ```X```,
            with ```X``` values scaled.
        
        
        '''
        X_out = copy.deepcopy(X)

        if 'labelled' in X_out:
            if self.group_name in X['labelled']:
                groups = X['labelled'][self.group_name]
            else:
                groups = None
            X_out['labelled']['X'] = self.scalar.transform(X_out['labelled']['X'], groups=groups)
        
        if 'unlabelled' in X_out:
            if self.group_name in X['unlabelled']:
                groups = X['unlabelled'][self.group_name]
            else:
                groups = None
            X_out['unlabelled']['X'] = self.scalar.transform(X_out['unlabelled']['X'], groups=groups)
        
        if 'X' in X_out:
            if self.group_name in X:
                groups = X[self.group_name]
            else:
                groups = None
            X_out['X'] = self.scalar.transform(X_out['X'], groups=groups)

        return X_out











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