from __future__ import annotations

import typing
import numpy as np
import sklearn
from aml.preprocessing.transformation_functions import flatten

class FlattenWrapper(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(
        self,
        estimator:typing.Any,
        start_dim:int=1,
        end_dim:int=-1,
        flatten_idx:int=0,
        unflatten_transform:bool=True,
        **estimator_kwargs,
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

        - `estimator_kwargs`: optional:
            The keywords that will be passed
            to the estimator init function.
            Defaults to `{}`.

        '''
        
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.flatten_idx = flatten_idx

        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.estimator_init = None

        self.unflatten_transform = unflatten_transform
        
        return
    
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
        self.estimator_init = self.estimator(**self.estimator_kwargs)
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