from __future__ import annotations
import numpy as np
import sklearn
import typing


from .utils import get_default_args

class SKModelWrapperDD(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, 
                    model:typing.Any, 
                    **kwargs,
                    ) -> None:
        '''
        A wrapper for any scikit-learn model
        that accepts a dictionary containing the 
        data. This is useful to use when combining
        semi-supervised methods with 
        ```sklearn.pipeline.Pipeline```.
        The model should not be initiated yet, and
        all arguments passed as positional or keyword
        arguments after the model is given.        
        
        Arguments
        ---------
        
        - ```model```: ```typing.Any```: 
            The model to wrap. This model must have
            both ```.fit(X, y)``` and ```.predict(X)```.
            An example would be an abstracted pytorch model.
        
        - ```*kwargs```: ```typing.Any```:
            Keyword arguments given to the model init.
        
        '''

        self._params_model = get_default_args(model)
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            self._params_model[key] = value

        self.model = model
        
        self._params = {}
        self._params['model'] = model
        self._params.update(**self._params_model)

        return
    
    def get_params(self, deep:bool=True) -> dict:
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
            if key in self._params_model:
                self._params_model[key] = value
            if key in self._params:
                self._params[key] = value
        return super().set_params(**params)

    def fit(self, 
            X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]], 
            y:None=None,
            ) -> SKModelWrapperDD:
        '''
        This will fit the model being wrapped.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]```: 
            A dictionary containing the data.
            For example, if ```semi_supervised=True```:
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
        
        - ```y```: ```None```, optional:
            Ignored. Please pass labels in the dictionary to 
            ```X```.
            Defaults to ```None```.
        
        
        
        Returns
        --------
        
        - ```self```: ```SKModelWrapperDD``` : 
            This object.
        
        
        '''

        self.model_init = self.model(**self._params_model)

        if 'labelled' in X:
            self.model_init.fit(X['labelled']['X'], X['labelled']['y'])

        if 'X' in X:
            self.model_init.fit(X['X'], X['y'])
        
        return self
    
    def predict(self, 
                X:typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]],
                return_ground_truth:bool=False,
                )->np.ndarray:
        '''
        This will predict using the model being wrapped.
        
        
        
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
        
        - ```return_ground_truth```: ```bool```, optional: 
            Whether to return the ground truth with the output.
            This is useful if this model was part of a pipeline
            in which the labels are altered.
        
        Returns
        --------
        
        - ```predictions```: ```numpy.ndarray``` : 
            The predictions, as a numpy array.
        
        - ```labels```: ```numpy.ndarray``` : 
            The labels, as a numpy array. Only returned
            if ```return_ground_truth=True```.
        

        '''
        if 'labelled' in X:
            output = self.model_init.predict(X['labelled']['X'])
            labels = X['labelled']['y']
        
        if 'X' in X:
            output = self.model_init.predict(X['X'])
            labels = X['y']
        
        if return_ground_truth:
            return output, labels

        return output
