from __future__ import annotations
from re import S
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
        :code:`sklearn.pipeline.Pipeline`.

        Note: Any attribute or method of the underlying transformer
        is accessible as normal as an attribute of this class.
        The returned value will be wrapped in a list if multiple
        :code:`fit_on` arguments are used, otherwise a single value is returned.
        
        
        Arguments
        ---------
        
        - transformer: typing.Any: 
            The transformer to wrap. This transformer must have
            both :code:`.fit(X, y)` and :code:`.transform(X)`.
            An example would be a custom transformer
            that uses data dictionaries.
        
        - fit_on: typing.Union[typing.List[str], typing.List[typing.List[str]]]: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the :code:`.fit()` function. Multiple inner lists
            will cause the :code:`.fit()` function to be called
            multiple times, with each fitted version being saved as
            a unique object in this class. If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one :code:`.fit()` is called, with arguments corresponding
            to the keys given as strings.
            The multiple fit models will be saved in a list, accessible
            through the :code:`fitted_models` attribute.
            Defaults to :code:`[['X', 'y']]`.

        - transform_on: typing.Union[typing.List[str], typing.List[typing.List[str]]]: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the :code:`.transform()` function. Multiple inner lists
            will cause the :code:`.transform()` function to be called
            multiple times, with each transform corresponding to the
            fitted object in each of the fit calls. If there are
            more :code:`.transform()` calls than :code:`.fit()` calls,
            then the transform will be called on the beginning of the fit
            object list again (ie: the transform calls  
            roll over the fit calls). The first key in each inner list
            will be overwritten with the result from :code:`.transform()`,
            unless :code:`all_key_transform=True`.
            If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one :code:`.transform()` is called, with arguments corresponding
            to the keys given as strings.
            Defaults to :code:`[['X']]`.
        
        - all_key_transform: bool, optional:
            This dictates whether the transformer being wrapped
            will output a result for all of the arguments given to 
            it. In this case, each of the values corresponding to the
            keys being transformed on, will be replaced by the 
            corresponding output of the wrapped transform. If :code:`False`,
            only the first key will be transformed.
            ie: :code:`x, y, z = self.wrapped_transform(x, y, z)`
            rather than: :code:`x = self.wrapped_transform(x, y, z)`
            Defaults to :code:`False`.

        - **kwargs: typing.Any:
            Keyword arguments given to the transformer.
        
        '''

        self._transformer_class = transformer
        self._transformer_init = self._transformer_class(**kwargs)
        self._params_transformer = kwargs

        if hasattr(self._transformer_init, 'get_params'):
            params_iterate = self._transformer_init.get_params()
        else:
            params_iterate = kwargs

        self._params_transformer_show = {}
        for key, value in params_iterate.items():
            if hasattr(value, 'get_params'):
                # if initiated
                try:
                    self._params_transformer_show.update(**value.get_params())
                    value_name = type(value).__name__
                except:
                    self._params_transformer_show.update(
                        **{k:v 
                            for k,v in get_default_args(value).items() 
                            if not k in kwargs
                            }
                        )
                    value_name = value.__name__
                self.__setattr__(key, value_name)
                self._params_transformer_show[key] = value_name
            else:
                self.__setattr__(key, value)
                self._params_transformer_show[key] = value


        self.fit_on = fit_on
        self.transform_on = transform_on
        self.all_key_transform = all_key_transform
        self._fitted_transformers = None

        self._params = {}
        self._params['transformer'] = self._transformer_init
        self._params['fit_on'] = self.fit_on
        self._params['transform_on'] = self.transform_on
        self._params['all_key_transform'] = self.all_key_transform
        
        self._params.update(**self._params_transformer_show)

        return
        
    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        if isinstance(self, type):
            return [
                'transformer', 
                'fit_on', 
                'transform_on', 
                'all_key_transform'
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
            return SKTransformerWrapperDD()._params
        
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
        
        - **params: dict`
            Estimator parameters.
        
        Returns
        ---------
        - self: estimator` instance
            Estimator instance.

        
        '''
        for key, value in params.items():
            if key in self._params_transformer_show:
                self._params_transformer[key] = value
                if hasattr(value, 'get_params'):
                    # if initiated
                    try:
                        self._params_transformer.update(**value.get_params())
                        value_name = type(value).__name__
                    except:
                        value_name = value.__name__
                    self._params_transformer_show[key] = value_name
                    params[key] = value_name
                else:
                    self._params_transformer_show[key] = value
                self._params.update(**self._params_transformer_show)
            if key in self._params:
                self._params[key] = value
            self._transformer_init = self._transformer_class(**self._params_transformer)
            self._params['transformer'] = self._transformer_init
        return super(SKTransformerWrapperDD, self).set_params(**params)

    def fit(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,
            ) -> SKTransformerWrapperDD:
        '''
        This will fit the transformer being wrapped.
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`fit_on` arguments will be ignored
            and the transformer will be passed :code:`.fit(X,y)`.
            In this case, consider using sklearn.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.
        
        - y: None, optional:
            Ignored unless :code:`X` is a :code:`numpy.ndarray`.
            If using a data dictionary, please pass labels 
            in the dictionary to :code:`X`.
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - self: SKTransformerWrapperDD: 
            This object.
        
        
        '''
        if not any(isinstance(i, list) for i in self.fit_on):
            fit_on_ = [self.fit_on]
        else:
            fit_on_ = self.fit_on

        self._fitted_transformers = []
        
        if type(X) == np.ndarray:
            transformer_init = self._transformer_class(**self._params_transformer)
            transformer_init.fit(X, y)
            self._fitted_transformers.append(transformer_init)
            return self

        for keys in fit_on_:
            transformer_init = self._transformer_class(**self._params_transformer)
            data = [X[key] for key in keys]
            transformer_init.fit(*data)
            self._fitted_transformers.append(transformer_init)

        return self
    
    def transform(self, 
                    X:typing.Dict[str, np.ndarray],
                    ) -> typing.Dict[str, np.ndarray]:
        '''
        This will transform the data using the transformer being wrapped.
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`transform_on` arguments will be ignored
            and the model will be passed :code:`.transform(X)`.
            In this case, consider using sklearn. In addition,
            this will be performed on the first fitted model 
            if many are fitted.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.
        
        
        Returns
        --------
        
        - X_out: typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]: 
            The data dictionary, with stucture the same as :code:`X`,
            with transformed :code:`X` data.
            If :code:`X` is a :code:`numpy.ndarray`, then a :code:`numpy.ndarray` will
            be returned.

        '''
        
        X_out = copy.deepcopy(X)

        if not any(isinstance(i, list) for i in self.transform_on):
            transform_on_ = [self.transform_on]
        else:
            transform_on_ = self.transform_on

        if self._fitted_transformers is None:
            raise TypeError('Please fit the trasform first.')

        if type(X) == np.ndarray:
            return self._fitted_transformers[0].transform(X)

        for nk, keys in enumerate(transform_on_):
            data = [X_out[key] for key in keys]
            outputs = (self._fitted_transformers[nk%len(self._fitted_transformers)]
                        .transform(*data))
            if self.all_key_transform:
                if len(data) == 1:
                    outputs = [outputs]
                for key, output in zip(keys, outputs):
                    X_out[key] = output
            else:
                X_out[keys[0]] = outputs

        return X_out

    def fit_transform(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,):
        '''
        This will fit and transform the transformer being wrapped 
        and the data given.
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`fit_on` arguments will be ignored
            and the transformer will be passed :code:`.fit_transform(X,y)`.
            In this case, consider using sklearn.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.
        
        - y: None, optional:
            Ignored unless :code:`X` is a :code:`numpy.ndarray`.
            If using a data dictionary, please pass labels 
            in the dictionary to :code:`X`.
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - X_out: typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]: 
            The data dictionary, with stucture the same as :code:`X`,
             with transformed :code:`X` data.
            If :code:`X` is a :code:`numpy.ndarray`, then a :code:`numpy.ndarray` will
            be returned.
        
        
        '''
        if hasattr(self._transformer_init, 'fit_transform'):
            if not any(isinstance(i, list) for i in self.fit_on):
                fit_on_ = [self.fit_on]
            else:
                fit_on_ = self.fit_on

            self._fitted_transformers = []
            
            if type(X) == np.ndarray:
                transformer_init = self._transformer_class(**self._params_transformer)
                out = transformer_init.fit_transform(X, y)
                self._fitted_transformers.append(transformer_init)
                return out

            else:
                X_out = copy.deepcopy(X)
                for keys in fit_on_:
                    transformer_init = self._transformer_class(**self._params_transformer)
                    data = [X[key] for key in keys]
                    outputs = transformer_init.fit_transform(*data)
                    self._fitted_transformers.append(transformer_init)
                    if self.all_key_transform:
                        if len(data) == 1:
                            outputs = [outputs]
                        for key, output in zip(keys, outputs):
                            X_out[key] = output
                    else:
                        X_out[keys[0]] = outputs
                return X_out
            
        else:
            self.fit(X=X, y=y)
            return self.transform(X=X)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if self._fitted_transformers is None:
            if hasattr(self._transformer_init, name):
                attr_list = [getattr(self._transformer_init, name)]
            else:
                raise AttributeError(f"{type(self).__name__} and "\
                    f"{type(self._transformer_init).__name__} have no attribute {name}")
        else:
            if hasattr(self._fitted_transformers[0], name):
                attr_list = [getattr(transformer, name) for transformer in self._fitted_transformers]
            else:
                raise AttributeError(f"{type(self).__name__} and "\
                    f"{type(self._fitted_transformers[0]).__name__} have no attribute {name}")
        
        if np.all([callable(attr) for attr in attr_list]):
            def wrapper(*args, **kwargs):
                return_list = [attr(*args, **kwargs) for attr in attr_list]
                if len(return_list) == 1:
                    return return_list[0]
                else:
                    return return_list
            return wrapper
        else:
            if len(attr_list) == 1:
                return attr_list[0]
            else:
                return attr_list

    @property
    def transformer(self,):
        """
        This is the transformer, which can be accessed as an attribute 
        of this class. If multiple :code:`fit_on` arguments were given,
        and the class has been fitted, then this will be a list
        of the fitted transformers. Otherwise, it will be a single instance.
        """
        if self._fitted_transformers is None:
            return self._transformer_init
        else:
            if len(self._fitted_transformers) == 1:
                return self._fitted_transformers[0]
            else:
                return self._fitted_transformers