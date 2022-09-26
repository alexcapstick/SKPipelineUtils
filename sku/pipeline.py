from __future__ import annotations
import copy
import typing
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
import warnings

class PipelineDD(Pipeline):
    def fit(self, 
            X:typing.Dict[str, np.ndarray], 
            y:None=None,
            *args, 
            **kwargs,
            ) -> PipelineDD:
        '''
        This will fit the pipeline.
        
        
        
        Arguments
        ---------
        
        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            When supplying :code:`X`, please be mindful
            of the objects in the pipeline and their 
            data requirements. If you are using model wrappers
            and transformer wrappers from this package, then 
            using a dictionary will be more powerful.
            However, you may supply a :code:`numpy.ndarray`,
            but all :code:`fit_on`, and :code:`transform_on`
            and :code:`predict_on` arguments will be ignored
            and sklearn defaults will be used.
            An example :code:`X`: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.
        
        - y: None, optional:
            Ignored unless :code:`X` is a :code:`numpy.ndarray`.
            If using a data dictionary, please pass labels 
            in the dictionary to :code:`X`.
            Defaults to :code:`None`.

        
        Returns
        --------
        
        - self: PipelineDD: 
            This object.
        
        
        '''

        return super().fit(X, y, *args, **kwargs,)


    def transform(
        self, 
        X:typing.Dict[str, np.ndarray],
        ) -> typing.Dict[str, np.ndarray]:

        X_copy = copy.deepcopy(X)
        passed_predict = False
        for _, name, layer in self._iter():
            if hasattr(layer, 'transform'):
                if passed_predict:
                    warnings.warn(f'A non-transform ({name}) has been passed, '\
                        'but the transforms have not been exhausted.')
                X_copy = layer.transform(X_copy)
            else:
                passed_predict = True

        return X_copy

    def predict(self, 
                X:typing.Dict[str, np.ndarray],
                *args, 
                **kwargs,
                ) -> typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]:
        '''
        This will predict using the fitted pipeline.
        
        
        
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

        
        Returns
        --------
        
        - predictions: numpy.ndarray: 
            The predictions, as a numpy array. If multiple
            inner lists are given as :code:`predict_on`, then
            a list of predictions will be returned.
        

        '''

        return super().predict(X, *args, **kwargs)


    def score(
        self, 
        X:typing.Dict[str, np.ndarray], 
        y:typing.Union[str, np.ndarray], 
        sample_weight=None,
        ):
        '''
        Transform the data, and apply :code:`score` with the final estimator.
        Call :code:`transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        :code:`score` method. Only valid if the final estimator implements :code:`score`.
        
        
        Arguments
        ---------

        - X: typing.Dict[str, np.ndarray]: 
            A dictionary containing the data.
            For example: :code:`X = {'X': X_DATA, 'y': Y_DATA, **kwargs}`.

        - y: typing.Union[str, np.ndarray]:
            Please either pass a string, which corresponds to the 
            key in :code:`X` which contains the labels, or pass
            the labels themselves.

        - sample_weight: array-like:
            If not None, this argument is passed as :code:``sample_weight`` keyword
            argument to the :code:``score`` method of the final estimator.
            Defaults to :code:`None`.
        
        
        Returns
        ---------
        - score: float`
            Result of calling :code:`score` on the final estimator.


        '''
        X_copy = copy.deepcopy(X)
        for _, name, transform in self._iter(with_final=False):
            X_copy = transform.transform(X_copy)
        
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        
        if type(y) == str:
            labels = X_copy[y]
        elif type(y) == np.ndarray:
            labels = y
        else:
            raise TypeError('y must be a string or numpy.ndarray.')
        
        return self.steps[-1][1].score(X_copy, labels, **score_params)



def pipeline_constructor(
    name:str, 
    name_to_object:typing.Dict[str, sklearn.base.BaseEstimator], 
    **pipeline_kwargs,
    ) -> PipelineDD:
    '''
    A function that constructs a pipeline from a string
    of pipeline keys and a dictionary mapping the keys to 
    pipeline objects.


    Example
    ---------
    .. code-block:: 

        >>> name_to_object = {
                            
                'standard_scaler': sku.SKTransformerWrapperDD(StandardScaler, 
                                    fit_on=['X'], 
                                    transform_on=['X'],
                                    ),

                'ae': sku.SKTransformerWrapperDD(
                        transformer=AEModel,
                        n_input=19, 
                        n_embedding=19, 
                        n_epochs=5,
                        n_layers=2,
                        dropout=0.2,
                    , optimizer={'adam':{'lr':0.01}},
                        criterion=nn.MSELoss(), 
                        fit_on=['X_unlabelled'],
                        transform_on =[['X'], 
                            ['X_unlabelled']],
                        ),

                'mlp': sku.SKModelWrapperDD(MLPModel,
                        n_input=19, 
                        n_output=2, 
                        hidden_layer_sizes=(100,50),
                        dropout=0.2,
                        n_epochs=10,
                    , optimizer={'adam':{'lr':0.01}},
                        criterion=nn.CrossEntropyLoss(), 
                        fit_on=['X', 'y'],
                        predict_on=['X'],
                        )

                }
        >>> name = 'standard_scaler--ae--standard_scaler--mlp'
        >>> pipeline = pipeline_constructor(name, name_to_object)


    Here, the pipeline will be returned as:
    .. code-block:: 

        [
            ['standard_scaler1', sku.SKTransformerWrapperDD],
            ['ae', sku.SKTransformerWrapperDD],
            ['standard_scaler2', sku.SKTransformerWrapperDD],
            ['mlp', sku.SKModelWrapperDD],
        ]

    Note the change in names of the standard scalers to 
    ensure that object names are unique.

    
    Arguments
    ---------
    
    - name: str: 
        A string in which the pipeline object names are
        split by :code:`'--'`. For example:
        :code:`'standard_scaler--ae--standard_scaler--mlp'`.
    
    - name_to_object: typing.Dict[str, sklearn.base.BaseEstimator]: 
        A dictionary mapping the strings in the :code:`name` argument
        to objects that will be placed in the pipeline. These objects
        must have a :code:`.fit()` and :code:`.predict()` method
        and use data dictionaries. Please see :code:`sku.SKTransformerWrapperDD`
        and :code:`sku.SKModelWrapperDD` for further explanation of structure.
    
    - pipeline_kwargs: 
        These are any extra keyword arguments that will be passed
        to :code:`sku.PipelineDD`.
    

    Returns
    --------
    
    - out: sku.pipeline.PipelineDD: 
        A data dictionary pipeline.
    
    
    '''

    objects_in_pipeline = name.split('--')
    names_in_pipeline = [v + str(objects_in_pipeline[:i].count(v) + 1) 
                            if objects_in_pipeline.count(v) > 1 
                            else v 
                            for i, v in enumerate(objects_in_pipeline)]
    try:
        pipeline = PipelineDD([[name, name_to_object[object]] 
                                            for name, object in zip(names_in_pipeline, objects_in_pipeline)],
                                        **pipeline_kwargs)
    except KeyError as e:
        raise type(e)(f'{str(e)} was not found in name_to_object.')

    return pipeline