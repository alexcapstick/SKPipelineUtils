from __future__ import annotations
import numpy as np
import sklearn
import copy
import typing

from .utils import get_default_args


class SKTransformerWrapperDD(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, 
                    transformer:typing.Any, 
                    fit_on:typing.Union[typing.List[str], typing.List[typing.List[str]]] = [['X', 'y']],
                    transform_on:typing.Union[typing.List[str], typing.List[typing.List[str]]] = [['X']],
                    all_key_transform=False,
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
        
        - ```fit_on```: ```typing.Union[typing.List[str], typing.List[typing.List[str]]]```: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the ```.fit()``` function. Multiple inner lists
            will cause the ```.fit()``` function to be called
            multiple times, with each fitted version being saved as
            a unique object in this class. If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one ```.fit()``` is called, with arguments corresponding
            to the keys given as strings.
            Defaults to ```[['X', 'y']]```.

        - ```transform_on```: ```typing.Union[typing.List[str], typing.List[typing.List[str]]]```: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the ```.transform()``` function. Multiple inner lists
            will cause the ```.transform()``` function to be called
            multiple times, with each transform corresponding to the
            fitted object in each of the fit calls. If there are
            more ```.transform()``` calls than ```.fit()``` calls,
            then the transform will be called on the beginning of the fit
            object list again (ie: the transform calls indefinitely 
            roll over the fit calls). The first key in each inner list
            will be overwritten with the result from ```.transform()```,
            unless ```all_key_transform=True```.
            If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one ```.transform()``` is called, with arguments corresponding
            to the keys given as strings.
            Defaults to ```[['X']]```.
        
        - ```all_key_transform```: ```bool```, optional:
            This dictates whether the transformer being wrapped
            will output a result for all of the arguments given to 
            it. In this case, each of the values corresponding to the
            keys being transformed on, will be replaced by the 
            corresponding output of the wrapped transform. If ```False```,
            only the first key will be transformed.
            ie: 
            ```
            x, y, z = self.wrapped_transform(x, y, z)
            ```
            rather than:
            ```
            x = self.wrapped_transform(x, y, z)
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
        self.transformer_init = self.transformer(**self._params_transformer)
        self.fit_on = fit_on
        self.transform_on = transform_on
        self.all_key_transform = all_key_transform
        self.fitted_transformers = None

        self._params = {}
        self._params['transformer'] = self.transformer_init
        self._params['fit_on'] = self.fit_on
        self._params['transform_on'] = self.transform_on
        self._params['all_key_transform'] = self.all_key_transform
        
        self._params.update(**self._params_transformer)

        return
        
    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        if isinstance(self, type):
            return ['transformer', 'fit_on', 'transform_on']
        else:
            parameters = list(self.get_params().keys())
            return sorted([p.name for p in parameters])
        
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
                self.transformer_init = self.transformer(**self._params_transformer)
                self._params.update(**self._params_transformer)
                self._params['transformer'] = self.transformer_init
            if key in self._params:
                self._params[key] = value
        return super(SKTransformerWrapperDD, self).set_params(**params)

    def fit(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,
            ) -> SKTransformerWrapperDD:
        '''
        This will fit the transformer being wrapped.
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, np.ndarray]```: 
            A dictionary containing the data.
            For example:
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
        if not any(isinstance(i, list) for i in self.fit_on):
            fit_on_ = [self.fit_on]
        else:
            fit_on_ = self.fit_on

        self.fitted_transformers = []
        for keys in fit_on_:
            transformer_init = self.transformer(**self._params_transformer)
            data = [X[key] for key in keys]
            transformer_init.fit(*data)
            self.fitted_transformers.append(transformer_init)

        return self
    
    def transform(self, 
                    X:typing.Dict[str, np.ndarray],
                    ) -> typing.Dict[str, np.ndarray]:
        '''
        This will transform the data using the transformer being wrapped.
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, np.ndarray]```: 
            A dictionary containing the data.
            For example:
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

        if not any(isinstance(i, list) for i in self.transform_on):
            transform_on_ = [self.transform_on]
        else:
            transform_on_ = self.transform_on

        if self.fitted_transformers is None:
            raise TypeError('Please fit the trasform first.')
        
        for nk, keys in enumerate(transform_on_):
            data = [X_out[key] for key in keys]
            outputs = (self.fitted_transformers[nk%len(self.fitted_transformers)]
                        .transform(*data))
            if self.all_key_transform:
                if len(data) == 1:
                    outputs = [outputs]
                for key, output in zip(keys, outputs):
                    X_out[key] = output
            else:
                X_out[keys[0]] = outputs

        return X_out