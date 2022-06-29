from __future__ import annotations
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
        
        - ```fit_on```: ```typing.Union[typing.List[str], typing.List[typing.List[str]]]```: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the ```.fit()``` function. Multiple inner lists
            will cause the ```.fit()``` function to be called
            multiple times. If a list of strings is given then
            they will be wrapped in an outer list, meaning that 
            one ```.fit()``` is called, with arguments corresponding
            to the keys given as strings.
            Defaults to ```[['X', 'y']]```.

        - ```predict_on```: ```typing.Union[typing.List[str], typing.List[typing.List[str]]]```: 
            This allows the user to define the keys in the data 
            dictionary that will be passed to the fit function.
            The outer list will be iterated over, and the inner
            list's keys will be used to get the data from the data dictionary,
            which will be passed in that order as positional arguments
            to the ```.predict()``` function. Multiple inner lists
            will cause the ```.predict()``` function to be called
            multiple times, with each predict corresponding to the
            fitted object in each of the fit calls. If there are
            more ```.predict()``` calls than ```.fit()``` calls,
            then the predict will be called on the beginning of the fit
            object list again (ie: the predict calls indefinitely 
            roll over the fit calls).
            Defaults to ```[['X']]```.
        
        - ```*kwargs```: ```typing.Any```:
            Keyword arguments given to the model init.
        
        '''
        self._params_model = get_default_args(model)
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            self._params_model[key] = value

        self.model = model
        self.fit_on = fit_on
        self.predict_on = predict_on
        self._params = {}
        self.model_init = self.model(**self._params_model)
        self.fitted_models = None

        self._params['model'] = self.model_init
        self._params['fit_on'] = self.fit_on
        self._params['predict_on'] = self.predict_on
        self._params.update(**self._params_model)

        return

    @classmethod
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        if isinstance(self, type):
            return ['model']
        else:
            parameters = list(self.get_params().keys())
            return sorted([p.name for p in parameters])
        
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
                self.model_init = self.model(**self._params_model)
                self._params.update(**self._params_model)
                self._params['model'] = self.model_init
            if key in self._params:
                self._params[key] = value
        return super(SKModelWrapperDD, self).set_params(**params)

    def fit(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,
            ) -> SKModelWrapperDD:
        '''
        This will fit the model being wrapped.
        
        
        
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
        
        - ```self```: ```SKModelWrapperDD``` : 
            This object.
        
        
        '''
        if not any(isinstance(i, list) for i in self.fit_on):
            fit_on_ = [self.fit_on]
        else:
            fit_on_ = self.fit_on

        self.fitted_models = []
        for keys in fit_on_:
            model_init = self.model(**self._params_model)
            data = [X[key] for key in keys]
            model_init.fit(*data)
            self.fitted_models.append(model_init)

        return self
    
    def predict(self, 
                X:typing.Dict[str, np.ndarray],
                return_data_dict:bool=False,
                ) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
        '''
        This will predict using the model being wrapped.
        
        
        
        Arguments
        ---------
        
        - ```X```: ```typing.Dict[str, np.ndarray]```: 
            A dictionary containing the data.
            For example:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.
        
        - ```return_data_dict```: ```bool```, optional: 
            Whether to return the ground truth with the output.
            This is useful if this model was part of a pipeline
            in which the labels are altered.
        
        Returns
        --------
        
        - ```predictions```: ```numpy.ndarray``` : 
            The predictions, as a numpy array. If multiple
            inner lists are given as ```predict_on```, then
            a list of predictions will be returned.
        
        - ```data_dict```: ```typing.Dict[str, np.ndarray]``` : 
            The labels, as a numpy array. Only returned
            if ```return_data_dict=True```.
        

        '''

        if self.fitted_models is None:
            raise TypeError('Please fit the model first.')

        output = []

        if not any(isinstance(i, list) for i in self.predict_on):
            predict_on_ = [self.predict_on]
        else:
            predict_on_ = self.predict_on

        for nk, keys in enumerate(predict_on_):
            data = [X[key] for key in keys]
            output.append(self.fitted_models[nk%len(self.fitted_models)]
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
        
        - ```X```: ```typing.Dict[str, np.ndarray]```: 
            A dictionary containing the data.
            For example:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.
        
        - ```return_data_dict```: ```bool```, optional: 
            Whether to return the ground truth with the output.
            This is useful if this model was part of a pipeline
            in which the labels are altered.
        
        Returns
        --------
        
        - ```predictions```: ```numpy.ndarray``` : 
            The predictions, as a numpy array. If multiple
            inner lists are given as ```predict_on```, then
            a list of predictions will be returned.
        
        - ```data_dict```: ```typing.Dict[str, np.ndarray]``` : 
            The labels, as a numpy array. Only returned
            if ```return_data_dict=True```.
        

        '''

        if self.fitted_models is None:
            raise TypeError('Please fit the model first.')

        output = []

        if not any(isinstance(i, list) for i in self.predict_on):
            predict_on_ = [self.predict_on]
        else:
            predict_on_ = self.predict_on

        for nk, keys in enumerate(predict_on_):
            data = [X[key] for key in keys]
            output.append(self.fitted_models[nk%len(self.fitted_models)]
                            .predict_proba(*data))

        if len(output) == 1:
            output = output[0]

        if return_data_dict:
            return output, X

        return output

