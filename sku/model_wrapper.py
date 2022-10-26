from __future__ import annotations
from unittest.mock import NonCallableMagicMock
import numpy as np
import sklearn
import typing


from .utils import get_default_args

class SKModelWrapperDD(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, 
                    model:typing.Any, 
                    fit_on:typing.Union[typing.List[str], typing.List[typing.List[str]]] = [['X', 'y']],
                    predict_on:typing.Union[typing.List[str], typing.List[typing.List[str]]] = [['X']],
                    **kwargs,
                    ) -> None:
        '''
        A wrapper for any scikit-learn model
        that accepts a dictionary containing the 
        data. This is useful to use when combining
        semi-supervised methods with 
        :code:`sklearn.pipeline.Pipeline`.
        The model should not be initiated yet, and
        all arguments passed as positional or keyword
        arguments after the model is given.        

        Note: Any attribute or method of the underlying model
        is accessible as normal as an attribute of this class.
        The returned value will be wrapped in a list if multiple
        :code:`fit_on` arguments are used, otherwise a single value is returned.
        
        Arguments
        ---------
        
        - model: typing.Any: 
            The model to wrap. This model must have
            both :code:`.fit(X, y)` and :code:`.predict(X)`.
            An example would be an abstracted pytorch model.
        
        - fit_on: typing.Union[typing.List[str], typing.List[typing.List[str]]]: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the :code:`.fit()` function. Multiple inner lists
            will cause the :code:`.fit()` function to be called
            multiple times. If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one :code:`.fit()` is called, with arguments corresponding
            to the keys given as strings.
            The multiple fit models will be saved in a list, accessible
            through the :code:`fitted_models` attribute.
            Defaults to :code:`[['X', 'y']]`.

        - predict_on: typing.Union[typing.List[str], typing.List[typing.List[str]]]: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the :code:`.predict()` function. Multiple inner lists
            will cause the :code:`.predict()` function to be called
            multiple times, with each predict corresponding to the
            fitted object in each of the fit calls. If there are
            more :code:`.predict()` calls than :code:`.fit()` calls,
            then the predict will be called on the beginning of the fit
            object list again (ie: the predict calls 
            roll over the fit calls).
            Defaults to :code:`[['X']]`.
        
        - *kwargs: typing.Any:
            Keyword arguments given to the model init.
        
        '''

        self._model_class = model
        self._model_init = self._model_class(**kwargs)
        self._params_model = kwargs

        if hasattr(self._model_init, 'get_params'):
            params_iterate = self._model_init.get_params()
        else:
            params_iterate = kwargs

        self._params_model_show = {}
        for key, value in params_iterate.items():
            if hasattr(value, 'get_params'):
                # if initiated
                try:
                    self._params_model_show.update(**value.get_params())
                    value_name = type(value).__name__
                except:
                    self._params_model_show.update(
                        **{k:v 
                            for k,v in get_default_args(value).items() 
                            if not k in kwargs
                            }
                        )
                    value_name = value.__name__
                self.__setattr__(key, value_name)
                self._params_model_show[key] = value_name
            else:
                self.__setattr__(key, value)
                self._params_model_show[key] = value


        self.fit_on = fit_on
        self.predict_on = predict_on
        self._fitted_models = None

        self._params = {}
        self._params['model'] = self._model_init
        self._params['fit_on'] = self.fit_on
        self._params['predict_on'] = self.predict_on

        self._params.update(**self._params_model_show)

        return

    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        if isinstance(self, type):
            return ['model']
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
            return SKModelWrapperDD()._params
        
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
            if key in self._params_model_show:
                self._params_model[key] = value
                if hasattr(value, 'get_params'):
                    # if initiated
                    try:
                        self._params_model.update(**value.get_params())
                        value_name = type(value).__name__
                    except:
                        value_name = value.__name__
                    self._params_model_show[key] = value_name
                    params[key] = value_name
                else:
                    self._params_model_show[key] = value
                self._params.update(**self._params_model_show)
            if key in self._params:
                self._params[key] = value
            self._model_init = self._model_class(**self._params_model)
            self._params['model'] = self._model_init
        return super(SKModelWrapperDD, self).set_params(**params)

    def fit(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,
            ) -> SKModelWrapperDD:
        '''
        This will fit the model being wrapped.
        
        
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`fit_on` arguments will be ignored
            and the model will be passed :code:`.fit(X,y)`.
            In this case, consider using sklearn.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.

        
        - y: None, optional:
            Ignored unless :code:`X` is a :code:`numpy.ndarray`.
            If using a data dictionary, please pass labels 
            in the dictionary to :code:`X`.
            Defaults to :code:`None`.
        
        
        
        Returns
        --------
        
        - self: SKModelWrapperDD: 
            This object.
        
        
        '''
        if not any(isinstance(i, list) for i in self.fit_on):
            fit_on_ = [self.fit_on]
        else:
            fit_on_ = self.fit_on

        self._fitted_models = []
        if type(X) == np.ndarray:
            model_init = self._model_class(**self._params_model)
            model_init.fit(X, y)
            self._fitted_models.append(model_init)
            return self

        for keys in fit_on_:
            model_init = self._model_class(**self._params_model)
            data = [X[key] for key in keys]
            model_init.fit(*data)
            self._fitted_models.append(model_init)

        return self
    
    def predict(self, 
                X:typing.Dict[str, np.ndarray],
                return_data_dict:bool=False,
                ) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
        '''
        This will predict using the model being wrapped.
        
        
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`predict_on` arguments will be ignored
            and the model will be passed :code:`.predict(X)`.
            In this case, consider using sklearn. In addition,
            this will be performed on the first fitted model 
            if many are fitted.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.

        
        - return_data_dict: bool, optional: 
            Whether to return the ground truth with the output.
            This is useful if this model was part of a pipeline
            in which the labels are altered. This is ignored
            if :code:`X` is a :code:`numpy.ndarray`
            Defaults to :code:`False`.

        
        Returns
        --------
        
        - predictions: numpy.ndarray: 
            The predictions, as a numpy array. If multiple
            inner lists are given as :code:`predict_on`, then
            a list of predictions will be returned.
        
        - data_dict: typing.Dict[str, np.ndarray]: 
            The labels, as a numpy array. Only returned
            if :code:`return_data_dict=True`.
        

        '''

        if self._fitted_models is None:
            raise TypeError('Please fit the model first.')

        output = []

        if not any(isinstance(i, list) for i in self.predict_on):
            predict_on_ = [self.predict_on]
        else:
            predict_on_ = self.predict_on

        if type(X) == np.ndarray:
            return self._fitted_models[0].predict(X)

        for nk, keys in enumerate(predict_on_):
            data = [X[key] for key in keys]
            output.append(self._fitted_models[nk%len(self._fitted_models)]
                            .predict(*data))

        if len(output) == 1:
            output = output[0]

        if return_data_dict:
            return output, X

        return output

    def predict_proba(self, 
                X:typing.Dict[str, np.ndarray],
                return_data_dict:bool=False,
                ) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
        '''
        This will predict the probabilities using the model being wrapped.
        
        
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            If :code:`X` is a :code:`numpy.ndarray`, then 
            the :code:`predict_on` arguments will be ignored
            and the model will be passed :code:`.predict_proba(X)`.
            In this case, consider using sklearn. In addition,
            this will be performed on the first fitted model 
            if many are fitted.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.
        
        - return_data_dict: bool, optional: 
            Whether to return the ground truth with the output.
            This is useful if this model was part of a pipeline
            in which the labels are altered. This is ignored
            if :code:`X` is a :code:`numpy.ndarray`
            Defaults to :code:`False`.

        
        Returns
        --------
        
        - predictions: numpy.ndarray: 
            The predictions, as a numpy array. If multiple
            inner lists are given as :code:`predict_on`, then
            a list of predictions will be returned.
        
        - data_dict: typing.Dict[str, np.ndarray]: 
            The labels, as a numpy array. Only returned
            if :code:`return_data_dict=True`.
        

        '''

        if self._fitted_models is None:
            raise TypeError('Please fit the model first.')

        output = []

        if not any(isinstance(i, list) for i in self.predict_on):
            predict_on_ = [self.predict_on]
        else:
            predict_on_ = self.predict_on

        if type(X) == np.ndarray:
            return self._fitted_models[0].predict_proba(X)

        for nk, keys in enumerate(predict_on_):
            data = [X[key] for key in keys]
            output.append(self._fitted_models[nk%len(self._fitted_models)]
                            .predict_proba(*data))

        if len(output) == 1:
            output = output[0]

        if return_data_dict:
            return output, X

        return output

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if self._fitted_models is None:
            if hasattr(self._model_init, name):
                attr_list = [getattr(self._model_init, name)]
            else:
                raise AttributeError(f"{type(self).__name__} and "\
                    f"{type(self._model_init).__name__} have no attribute {name}")
        else:
            if hasattr(self._fitted_models[0], name):
                attr_list = [getattr(model, name) for model in self._fitted_models]
            else:
                raise AttributeError(f"{type(self).__name__} and "\
                    f"{type(self._fitted_models[0]).__name__} have no attribute {name}")
        
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
    def model(self,):
        """
        This is the model, which can be accessed as an attribute 
        of this class. If multiple :code:`fit_on` arguments were given,
        and the class has been fitted, then this will be a list
        of the fitted models. Otherwise, it will be a single instance.
        """
        if self._fitted_models is None:
            return self._model_init
        else:
            if len(self._fitted_models) == 1:
                return self._fitted_models[0]
            else:
                return self._fitted_models
