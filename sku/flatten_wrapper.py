from __future__ import annotations
from re import L
import typing
import numpy as np
import sklearn

import aml

flatten = aml.flatten

from .utils import get_default_args



class FlattenWrapper(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(
        self,
        estimator:typing.Any=None,
        start_dim:int=1,
        end_dim:int=-1,
        flatten_arg:typing.Union[int, str]=0,
        unflatten_transform:bool=True,
        **kwargs,
        ) -> None:
        '''
        This class allows you to wrap any transformer or model with 
        a flattening operation. By default, the flattening
        will allow for shape (in_shape[0], -1). Please see the 
        flattening operations in 
        :code:`aml.flatten` to 
        understand the arguments :code:`start_dim` and :code:`end_dim`.
        
        Note: Any attribute of the underlying estimator is accessible as normal
        as an attribute of this class. If you require a flattened call
        of any method of the underlying estimator, then you may use a
        prefix of :code:`flatten__` to the method name. See the examples.

        
        
        Examples
        ---------
        .. code-block:: 

            >>> flatten_scaler = FlattenWrapper(
                                    StandardScaler, 
                                    unflatten_transform=True,
                                    )
            >>> flatten_scaler.fit(x)
            >>> flatten_scaler.transform(x)
            
        Also, since any attribute of the underlying estimator
        is accessible through this wrapper, you may do the 
        following:

        .. code-block:: 

            >>> flatten_scaler = FlattenWrapper(
                                    StandardScaler, 
                                    unflatten_transform=True,
                                    )
            >>> flatten_scaler.fit(x)
            >>> flatten_scaler.transform(x)
            >>> flatten_scaler.mean_
            [0.5, 0.2, 100, 20]

        If you require a function call of the
        underlying estimator, but still want to
        flatten the argument first. Use the following
        structure:

        .. code-block:: 

            >>> flatten_lr = FlattenWrapper(
                                    LogisticRegression, 
                                    )
            >>> flatten_lr.fit(x, y)
            >>> flatten_lr.flatten__predict_log_proba(x)
            [0, 1, 0, 1, 1, 0]



        Arguments
        ---------
        
        - estimator: typing.Any:
            The transformer or model that
            requires a flattening of the
            before fit and transform or predict.
            Defaults to :code:`None`.

        - start_dim: int, optional:
            The first dim to flatten. 
            Defaults to :code:`0`.
        
        - end_dim: int, optional:
            The last dim to flatten. 
            Defaults to :code:`-1`.
        
        - flatten_arg: int, optional:
            The argument of the call on fit
            and transform/predict that contains
            the array that requires flattening.
            This can either be the keyword, that 
            needs to be given in the method calls,
            or the idx of the argument.
            Defaults to :code:`0`.
        
        - unflatten_transform: bool, optional:
            Return an unflattened version of the
            transformed array. If the output
            is a numpy array, then this will 
            unflattened directly. If the output
            is a list or tuple, then the :code:`flatten_arg` of
            the output will be unflattened.
            Defaults to :code:`True`.

        - **kwargs:, optional:
            The keywords that will be passed
            to the estimator init function.
            Defaults to :code:`{}`.

        '''

        self.estimator_class = estimator
        if not estimator is None:
            self._estimator = self.estimator_class(**kwargs)
            if hasattr(self._estimator, 'get_params'):
                self._params_estimator = self._estimator.get_params()
                self._params_estimator.update(**kwargs)
            else:
                self._params_estimator = get_default_args(estimator)
                self._params_estimator.update(**kwargs)
        else:
            self._estimator = None
            self._params_estimator = {}

        for key, value in self._params_estimator.items():
            self.__setattr__(key, value)

        self.start_dim = start_dim
        self.end_dim = end_dim
        self.flatten_arg = flatten_arg
        self.unflatten_transform = unflatten_transform

        self._params = {}
        self._params['estimator'] = self._estimator
        self._params['start_dim'] = self.start_dim
        self._params['end_dim'] = self.end_dim
        self._params['flatten_arg'] = self.flatten_arg
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
                'flatten_arg',
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
        
        - deep: bool, optional:
            Ignored. 
            Defaults to :code:`True`.
        
        
        
        Returns
        --------
        
        - out: dict: 
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
        parameters of the form :code:``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        
        Arguments
        ---------
        
        - **params: dict:
            Estimator parameters.
        
        Returns
        ---------
        - self: estimator instance
            Estimator instance.

        
        '''
        for key, value in params.items():
            if key in self._params_estimator:
                self._params_estimator[key] = value
                self._estimator = self.estimator_class(**self._params_estimator)
                self._params.update(**self._params_estimator)
                self._params['estimator'] = self._estimator
            if key in self._params:
                self._params[key] = value
        return super(FlattenWrapper, self).set_params(**params)

    def _flatten_args_kwargs(self, *args, return_shape=False, **kwargs):

        args = list(args)
        kwargs = dict(kwargs)
        if type(self.flatten_arg) == int:
            if return_shape:
                in_shape = args[self.flatten_arg].shape
            args[self.flatten_arg] = flatten(
                args[self.flatten_arg], 
                start_dim=self.start_dim, 
                end_dim=self.end_dim, 
                copy=False,
                )

        elif type(self.flatten_arg) is str:
            if return_shape:
                in_shape = kwargs[self.flatten_arg].shape
            kwargs[self.flatten_arg] = flatten(
                kwargs[self.flatten_arg], 
                start_dim=self.start_dim, 
                end_dim=self.end_dim, 
                copy=False,
                )

        else:
            raise TypeError("Please ensure that flatten_arg is an int or str.")

        if return_shape:
            return args, kwargs, in_shape
        else:
            return args, kwargs

    def fit(
        self, 
        *args,
        **kwargs
        ) -> FlattenWrapper:
        '''
        Fit the estimator with the flattened array.


        Arguments
        ---------
        
        - args:
            Arguments passed to the estimator when fitting.
        
        - kwargs:
            Keyword arguments passed to the estimator when fitting.


        Returns
        --------
        
        - self: FlattenWrapper:
            The fitted estimator.
        
        
        '''
        
        assert not self._estimator is None, \
            "No estimator given."

        args, kwargs = self._flatten_args_kwargs(*args, **kwargs)

        self._estimator = self.estimator_class(**self._params_estimator)
        self._estimator.fit(
            *args
            )

        return self


    def _transform(self, 
                    *args,
                    **kwargs,
                    ) -> np.ndarray:
        '''
        Transform with the estimator and flattened array.


        Arguments
        ---------
        
        - args:
            Arguments passed to the estimator when transforming.
        
        - kwargs:
            Keyword arguments passed to the estimator when transforming.


        Returns
        --------
        
        - args_out: FlattenWrapper:
            The transformed input.
        
        
        '''

        args, kwargs, in_shape = self._flatten_args_kwargs(*args, **kwargs, return_shape=True)

        out = self._estimator.transform(
            *args,
            **kwargs,
            )

        # if transform output should be unflattened
        if self.unflatten_transform:

            if type(out) == np.ndarray:
                out = out.reshape(*in_shape)

            elif (type(out) == list) or (type(out) == tuple):
                if type(self.flatten_arg) == int:
                    out[self.flatten_arg] = out[self.flatten_arg].reshape(*in_shape)
                else:
                    raise TypeError("unflatten and a string flatten_arg is not " \
                        "compatible with a transformer that returns lists or tuples. "\
                        "Please use flatten_arg int instead.")
            elif type(out) == dict:
                if type(self.flatten_arg) == str:
                    out[self.flatten_arg] = out[self.flatten_arg].reshape(*in_shape)
                else:
                    raise TypeError("unflatten and an int flatten_arg is not " \
                        "compatible with a transformer that returns a dictionary. "\
                        "Please use flatten_arg str instead.")

        return out


    def _predict(self, 
                    *args,
                    **kwargs,
                    ) -> np.ndarray:
        '''
        Predict with the estimator and flattened array.


        Arguments
        ---------
        
        - args:
            Arguments passed to the estimator when predicting.
        
        - kwargs:
            Keyword arguments passed to the estimator when predicting.


        Returns
        --------
        
        - args_out: FlattenWrapper:
            The prediction input.
        
        
        '''
        
        
        assert not self._estimator is None, \
            "Please fit this class before calling the predict method."

        args, kwargs = self._flatten_args_kwargs(*args, **kwargs)

        out = self._estimator.predict(
            *args,
            **kwargs
            )

        return out


    def _predict_proba(self, 
                    *args,
                    **kwargs,
                    ) -> np.ndarray:
        '''
        Predict probability with the estimator and flattened array.


        Arguments
        ---------
        
        - args:
            Arguments passed to the estimator when predicting.
        
        - kwargs:
            Keyword arguments passed to the estimator when predicting.


        Returns
        --------
        
        - args_out: FlattenWrapper:
            The probability prediction input.
        
        
        '''
        
        assert not self._estimator is None, \
            "Please fit this class before calling the predict_proba method."

        args, kwargs = self._flatten_args_kwargs(*args, **kwargs)

        out = self._estimator.predict_proba(
            *args,
            **kwargs
            )

        return out

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):

        # whether the estimator is a model or transformer
        if name in ['transform', 'predict', 'predict_proba']:
            if hasattr(self._estimator, name):
                return getattr(self, '_'+name)

        # if the attr exists in the estimator
        if hasattr(self._estimator, name):
            attr = getattr(self._estimator, name)
            return attr
        # if the attr exists in the estimator
        # and it is prefixed with flatten__
        # then return a wrapper that flattens the
        # function first
        elif "flatten__" in name:
            name = name.replace("flatten__", "")
            if hasattr(self._estimator, name):
                attr = getattr(self._estimator, name)
                def wrapper(*args, **kwargs):
                    args = list(args)
                    args, kwargs = self._flatten_args_kwargs(*args, **kwargs)
                    return attr(*args, **kwargs)
                return wrapper
        # if the attr doesn't exist
        else:
            raise AttributeError(f"{type(self).__name__} and "\
                f"{type(self._estimator).__name__} have no attribute {name}")

    @property
    def estimator(self,):
        """
        This is the estimator, which can be accessed as an attribute 
        of this class.
        """
        return self._estimator
