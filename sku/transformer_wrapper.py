from __future__ import annotations
import numpy as np
import sklearn
import copy
import typing

from .utils import get_default_args


class SKTransformerWrapperDD(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, 
                    transformer:typing.Any, 
                    semi_supervised:bool=False,
                    **kwargs,
                    )->None:
        '''
        A wrapper for any scikit-learn transformer
        that accepts a dictionary containing the 
        data. This is useful to use when combining
        semi-supervised methods with 
        ```sklearn.pipeline.Pipeline```.
        
        
        Arguments
        ---------
        
        - ```transformer```: ```typing.Any```: 
            The model to wrap. This model must have
            both ```.fit(X, y)``` and ```.transform(X)```.
            An example would be a custom transformer
            that uses data dictionaries.
        
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
        
        - ```*kwargs```: ```typing.Any```:
            Keyword arguments given to the model.
        
        '''

        self._params_transformer = get_default_args(transformer)
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            self._params_transformer[key] = value

        self.transformer = transformer
        self.semi_supervised = semi_supervised
        
        self._params = {}
        self._params['transformer'] = transformer
        self._params['semi_supervised'] = semi_supervised
        
        self._params.update(**self._params_transformer)

        return
    
    def get_params(self,deep=True) -> dict:
        '''
        Overrides sklearn function.
        
        
        
        Arguments
        ---------
        
        - ```deep```: ```bool```, optional:
            Ignored. 
            Defaults to ```True```.
        
        
        
        Returns
        --------
        
        - ```out```: ```dict``` : 
            Dictionary of parameters.
        
        
        '''
        return self._params
    
    def set_params(self, **params):
        '''
        
        From sklearn documentation:
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        
        Arguments
        ---------
        
        - ```**params``` : ```dict```
            Estimator parameters.
        
        Returns
        ---------
        - ```self``` : ```estimator``` instance
            Estimator instance.

        
        '''
        for key, value in params.items():
            if key in self._params_transformer:
                self._params_transformer[key] = value
            if key in self._params:
                self._params[key] = value
        return super().set_params(**params)

    def fit(self, 
            X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]], 
            y:None=None,
            ) -> SKTransformerWrapperDD:
        '''
        This will fit the transformer being wrapped.
        
        
        
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
        
        - ```y```: ```None```, optional:
            Ignored. Please pass labels in the dictionary to 
            ```X```.
            Defaults to ```None```.
        
        
        Returns
        --------
        
        - ```self```: ```SKTransformerWrapperDD``` : 
            This object.
        
        
        '''
        self.transformer_init = self.transformer(**self._params_transformer)

        if self.semi_supervised:
            self.transformer_init.fit(X['unlabelled']['X'], X['unlabelled']['y'])
        else:
            self.transformer_init.fit(X['X'], X['y'])
        return self
    
    def transform(self, 
                    X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]],
                    ) -> typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]:
        '''
        This will transform the data using the transformer being wrapped.
        
        
        
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
            The data dictionary, with stucture the same as ```X```,
             with transformed ```X``` data.
        

        '''
        
        X_out = copy.deepcopy(X)

        if 'labelled' in X_out:
            X_out['labelled']['X'] = self.transformer_init.transform(X_out['labelled']['X'])
        
        if 'unlabelled' in X_out:
            X_out['unlabelled']['X'] = self.transformer_init.transform(X_out['unlabelled']['X'])
        
        if 'X' in X_out:
            X_out['X'] = self.transformer_init.transform(X_out['X'])

        return X_out