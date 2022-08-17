from __future__ import annotations
import typing
import numpy as np
import sklearn
from aml.preprocessing.transformation_functions import flatten

from .utils import get_default_args



class FlattenWrapper(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(
        self,
        estimator:typing.Any=None,
        start_dim:int=1,
        end_dim:int=-1,
        flatten_idx:int=0,
        unflatten_transform:bool=True,
        **kwargs,
        ) -> None:
        '''
        This class allows you to wrap any transformer or model with 
        a flattening operation. By default, the flattening
        will allow for shape (in_shape[0], -1). Please see the 
        flattening operations in 
        `aml.preprocessing.transformation_functions.flatten` to 
        understand the arguments `start_dim` and `end_dim`.

        
        
        Examples
        ---------
        ```
        >>> flatten_scaler = FlattenWrapper(
                                StandardScaler, 
                                unflatten_transform=True,
                                )
        >>> flatten_scaler.fit(x)
        >>> flatten_scaler.transform(x)
        ```



        Arguments
        ---------
        
        - `estimator`: `typing.Any`:
            The transformer or model that
            requires a flattening of the
            before fit and transform or predict.
            Defaults to `None`.

        - `start_dim`: `int`, optional:
            The first dim to flatten. 
            Defaults to `0`.
        
        - `end_dim`: `int`, optional:
            The last dim to flatten. 
            Defaults to `-1`.
        
        - `flatten_idx`: `int`, optional:
            The index of the call on fit
            and transform/predict that contains
            the array that requires flattening.
            Defaults to `0`.
        
        - `unflatten_transform`: `bool`, optional:
            Return an unflattened version of the
            transformed array. If the output
            is a numpy array, then this will 
            unflattened directly. If the output
            is a list or tuple, then the `flatten_idx` of
            the output will be unflattened.
            Defaults to `True`.

        - `**kwargs`: optional:
            The keywords that will be passed
            to the estimator init function.
            Defaults to `{}`.

        '''

        self.estimator = estimator
        if not estimator is None:
            self.estimator_init = self.estimator(**kwargs)
            if hasattr(self.estimator_init, 'get_params'):
                self._params_estimator = kwargs
                self._params_estimator.update(**self.estimator_init.get_params())
            else:
                self._params_estimator = kwargs
                self._params_estimator.update(**get_default_args(estimator))
        else:
            self.estimator_init = None
            self._params_estimator = {}

        for key, value in self._params_estimator.items():
            self.__setattr__(key, value)

        self.start_dim = start_dim
        self.end_dim = end_dim
        self.flatten_idx = flatten_idx
        self.unflatten_transform = unflatten_transform

        self._params = {}
        self._params['estimator'] = self.estimator_init
        self._params['start_dim'] = self.start_dim
        self._params['end_dim'] = self.end_dim
        self._params['flatten_idx'] = self.flatten_idx
        self._params['unflatten_transform'] = self.unflatten_transform
        
        self._params.update(**self._params_estimator)

        return

    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        if isinstance(self, type):
            return [
                'estimator', 
                'start_dim', 
                'end_dim',
                'flatten_idx',
                'unflatten_transform',
                ]
        else:
            parameters = list(self.get_params().keys())
            return sorted([p.name for p in parameters])


    def get_params(self=None, deep=True) -> dict:
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

        if not self is None:
            return self._params
        else:
            return FlattenWrapper()._params


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
            if key in self._params_estimator:
                self._params_estimator[key] = value
                self.estimator_init = self.estimator(**self._params_estimator)
                self._params.update(**self._params_estimator)
                self._params['estimator'] = self.estimator_init
            if key in self._params:
                self._params[key] = value
        return super(FlattenWrapper, self).set_params(**params)


    def fit(
        self, 
        *args,
        ) -> FlattenWrapper:
        '''
        Fit the estimator with the flattened array.


        Arguments
        ---------
        
        - `args`:
            Arguments passed to the estimator when fitting.


        Returns
        --------
        
        - `self`: `FlattenWrapper`:
            The fitted estimator.
        
        
        '''
        args = list(args)
        args[self.flatten_idx] = flatten(
                                    args[self.flatten_idx], 
                                    start_dim=self.start_dim, 
                                    end_dim=self.end_dim, 
                                    copy=False,
                                    )
        self.estimator_init = self.estimator(**self._params_estimator)
        self.estimator_init.fit(
            *args
            )

        return self


    def transform(self, 
                    *args,
                    ) -> np.ndarray:
        '''
        Transform with the estimator and flattened array.


        Arguments
        ---------
        
        - `args`:
            Arguments passed to the estimator when transforming.


        Returns
        --------
        
        - `args_out`: `FlattenWrapper`:
            The transformed input.
        
        
        '''
        
        
        assert not self.estimator_init is None, \
            "Please fit this class before calling the transform method."

        args = list(args)

        in_shape = args[self.flatten_idx].shape

        args[self.flatten_idx] = flatten(
                                    args[self.flatten_idx], 
                                    start_dim=self.start_dim, 
                                    end_dim=self.end_dim, 
                                    copy=False,
                                    )
        out = self.estimator_init.transform(
            *args
            )
        
        # if transform output should be unflattened
        if self.unflatten_transform:
            if type(out) == np.ndarray:
                out = out.reshape(*in_shape)

            elif (type(out) == list) or (type(out) == tuple):
                out[self.flatten_idx] = out[self.flatten_idx].reshape(*in_shape)

        return out


    def predict(self, 
                    *args,
                    ) -> np.ndarray:
        '''
        Predict with the estimator and flattened array.


        Arguments
        ---------
        
        - `args`:
            Arguments passed to the estimator when predicting.


        Returns
        --------
        
        - `args_out`: `FlattenWrapper`:
            The prediction input.
        
        
        '''
        
        
        assert not self.estimator_init is None, \
            "Please fit this class before calling the predict method."

        args = list(args)

        args[self.flatten_idx] = flatten(
                                    args[self.flatten_idx], 
                                    start_dim=self.start_dim, 
                                    end_dim=self.end_dim, 
                                    copy=False,
                                    )
        out = self.estimator_init.predict(
            *args
            )

        return out


    def predict_proba(self, 
                    *args,
                    ) -> np.ndarray:
        '''
        Predict probability with the estimator and flattened array.


        Arguments
        ---------
        
        - `args`:
            Arguments passed to the estimator when predicting.


        Returns
        --------
        
        - `args_out`: `FlattenWrapper`:
            The probability prediction input.
        
        
        '''
        
        
        assert not self.estimator_init is None, \
            "Please fit this class before calling the predict_proba method."

        args = list(args)

        args[self.flatten_idx] = flatten(
                                    args[self.flatten_idx], 
                                    start_dim=self.start_dim, 
                                    end_dim=self.end_dim, 
                                    copy=False,
                                    )
        out = self.estimator_init.predict_proba(
            *args
            )

        return out