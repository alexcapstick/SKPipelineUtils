import numpy as np
import pandas as pd
import typing
import tqdm
import uuid
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from .pipeline import PipelineDD, pipeline_constructor
from .progress import tqdm_style





class PipelineSearchCV(BaseEstimator):
    def __init__(self,
                    pipeline_names:typing.List[str],
                    name_to_object:typing.Dict[str, BaseEstimator],
                    metrics:typing.Dict[str, typing.Callable],
                    cv=None,
                    split_fit_on:typing.List[str]=['X', 'y'],
                    split_transform_on:typing.List[str]=['X', 'y'],
                    param_grid:typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]=None,
                    verbose:bool=False,
                    ):
        '''
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in 
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines. A parameter grid can also be passed, allowing
        the user to test multiple configurations of each pipeline.

        Example
        ---------

        ```
        name_to_object = {
                            'gbt': sku.SKModelWrapperDD(HistGradientBoostingClassifier,
                                                        fit_on=['X', 'y'],
                                                        predict_on=['X'],
                                                        ),
                            'standard_scaler': sku.SKTransformerWrapperDD(StandardScaler, 
                                                        fit_on=['X'], 
                                                        transform_on=['X'],
                                                        ),
                            }
        pipeline_names = [
                            'standard_scaler--gbt',
                            'gbt'
                        ]
        metrics = {
                    'accuracy': accuracy_score, 
                    'recall': recall_score, 
                    'precision': precision_score,
                    'f1': f1_score,
                    }
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1024)

        pscv = PipelineSearchCV(pipeline_names=pipeline_names,
                                    name_to_object=name_to_object,
                                    metrics=metrics,
                                    cv=splitter,
                                    split_fit_on=['X', 'y'],
                                    split_transform_on=['X', 'y', 'id'],
                                    verbose=True,
                                    param_grid={
                                                'gbt__learning_rate':[0.1, 0.01],
                                                'gbt__max_depth':[3, 10],
                                                },
                                    )
        X_data = {
                    'X': X_labelled, 'y': y_labelled, 'id': id_labelled,
                    'X_unlabelled': X_unlabelled, 'id_unlabelled': id_unlabelled,
                    }
        results = pscv.fit(X_data)
        ```


        
        Arguments
        ---------

        - ```pipeline_names```: ```typing.List[str]```:
            This is a list of strings that describe the pipelines
            An example would be ```standard_scaler--ae--mlp```.
            The objects, separated by '--' should be keys in 
            ```name_to_object```.
        
        - ```name_to_object```: ```typing.Dict[str, BaseEstimator]```:
            A dictionary mapping the keys in ```pipeline_names``` to
            the objects that will be used as transformers and 
            models in the pipeline.
        
        - ```metrics```: ```typing.Dict[str, typing.Callable]```:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            ```func(labels, predictions)```.
        
        - ```cv```: sklearn splitting class:
            This is the class that is used to produce the cross
            validation data. It should have the method
            ```.split(X, y, event)```, which returns the indices
            of the training and testing set, and the method 
            ```get_n_splits()```, which should return the number
            of splits that this splitter was indended to make.
            Defaults to ```None```.

        - ```split_fit_on```: ```typing.List[str]```:
            The keys corresponding to the values in 
            the data dictionary passed in ```.fit()```
            that the ```cv``` will take as positional
            arguments to the ```split()``` function.
            Defaults to ```['X', 'y']```.        

        - ```split_transform_on```: ```typing.List[str]```:
            The keys corresponding to the values in 
            the data dictionary passed in ```.fit()```
            that the ```cv``` will split into training 
            and testing data. This allows you to 
            split data that isn't used in finding the
            splitting indices.
            Defaults to ```['X', 'y']```.        

        - ```param_grid```: ```typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]```:
            A dictionary or list of dictionaries that 
            are used as a parameter grid for testing performance
            of pipelines with multiple hyper-parameters. This should
            be of the usual format to 
            ```sklearn.model_selection.ParameterGrid``` when 
            used with ```sklearn.pipeline.Pipeline```.
            If ```None```, then the pipeline is tested with 
            the parameters given to the objects in 
            ```name_to_object```.
            Defaults to ```None```.        
        
        - ```verbose```: ```bool```:
            Whether to print progress as the models are being tested.
            Remeber that you might also need to change the verbose options
            in each of the objects given in ```name_to_object```.
            Defaults to ```False```.     
        
        
        '''
        assert not cv is None, 'Currently cv=None is not supported. '\
                                'Please pass an initialised sklearn splitter.'

        self.pipeline_names = pipeline_names
        self.name_to_object = name_to_object
        self.metrics = metrics
        self.cv = cv
        self.verbose = verbose
        self.param_grid = param_grid
        self.split_fit_on = split_fit_on
        self.split_transform_on = split_transform_on
        self.param_grid = [None] if param_grid is None else ParameterGrid(param_grid)

        return

    @staticmethod
    def _test_single_split(
                            pipeline:PipelineDD,
                            train_data:typing.Dict[str, np.ndarray],
                            test_data:typing.Dict[str, np.ndarray],
                            ns:int,
                            metrics:typing.Dict[str, typing.Callable],
                            )->typing.List[typing.Dict[str,typing.Union[int, float]]]:
        '''
        Testing the whole pipeline, on a single train and test data.
        '''

        pipeline.fit(train_data)
        _, out_data = pipeline.predict(train_data, return_data_dict=True)
        train_y_out = out_data['y']

        predictions, out_data = pipeline.predict(test_data, return_data_dict=True)
        labels = out_data['y']
        results_temp = [
                        {
                        'metric': metric, 
                        'value': func(labels, predictions),
                        'split_number': ns,
                        #'train_positve': np.sum(train_y_out)/train_y_out.shape[0],
                        #'test_positve': np.sum(labels)/labels.shape[0],
                        } 
                        for metric, func in metrics.items()
                        ]
        
        return results_temp
    

    def _test_pipeline(self,
                    X,
                    pipeline_name,
                    pipeline_update_params=None,
                    ):
        '''
        Testing the whole pipeline, over the splits, with given params.
        '''

        pipeline = pipeline_constructor(pipeline_name,
                    name_to_object=self.name_to_object,
                    verbose=False)

        if not pipeline_update_params is None:
            pipeline_params = pipeline.get_params()
            pipeline_update_params = {k:v for k,v in pipeline_update_params.items() if k in pipeline_params}
            if len(pipeline_update_params)!=0:
                pipeline.set_params(**pipeline_update_params)
            else:
                results_temp = {
                                'pipeline': pipeline_name,
                                'metrics': [],
                                'splitter': type(self.cv).__name__,
                                'params': pipeline.get_params(),
                                'param_updates': None,
                                'train_id': uuid.uuid4(),
                                }
                self.tqdm_progress.update(self.cv.get_n_splits())
                return results_temp

        results_temp = {
                        'pipeline': pipeline_name,
                        'metrics': [],
                        'splitter': type(self.cv).__name__,
                        'params': pipeline.get_params(),
                        'param_updates': pipeline_update_params,
                        'train_id': uuid.uuid4(),
                        }

        for ns, (train_idx, test_idx) in enumerate(self.cv.split(*[ X[split_data] 
                                                    for split_data in self.split_fit_on ])):
            # data to split on
            train_data = { split_data:X[split_data][train_idx] for split_data in self.split_transform_on }
            test_data = { split_data:X[split_data][test_idx] for split_data in self.split_transform_on }
            # other data
            train_data.update({ k: v for k, v in X.items() if k not in train_data })
            test_data.update({ k: v for k, v in X.items() if k not in test_data })

            # fitting and predicting
            results_temp['metrics'].extend(self._test_single_split(
                                            pipeline=pipeline,
                                            train_data=train_data,
                                            test_data=test_data,
                                            ns=ns,
                                            metrics=self.metrics,
                                            ))

            self.tqdm_progress.update(1)

        return results_temp

    def _grid_test_pipeline(self,
                            X,
                            pipeline_name,
                            ):
        '''
        Testing the whole pipeline, over the splits and params.
        '''

        results_pipeline = pd.DataFrame()

        for g in self.param_grid:
            results_temp = self._test_pipeline(
                                                X=X,
                                                pipeline_name=pipeline_name,
                                                pipeline_update_params=g,
                                                )
            results_pipeline = pd.concat([
                                    results_pipeline, 
                                    pd.json_normalize(results_temp, 
                                                        record_path='metrics', 
                                                        meta=['pipeline', 
                                                                'splitter', 
                                                                'params', 
                                                                'train_id',
                                                                'param_updates',
                                                                ])
                                    
                                    ])

        return results_pipeline

    def fit(self,
            X:typing.Dict[str, np.ndarray],
            y=None,
            ):
        '''
        This function fits and predicts the pipelines, 
        with the optional parameters and splitting 
        arguments and produces a table of results
        given the metrics.

        Arguments
        ---------

        - ```X```: ```typing.Dict[str, np.ndarray]```:
            The data dictionary that will be used to run
            the experiments.
        
        - ```y```: ```None```:
            Ignored.
        
        
        
        '''

        self.tqdm_progress = tqdm.tqdm(
                                        total=len(self.pipeline_names)*self.cv.get_n_splits()*len(self.param_grid), 
                                        desc='Searching', 
                                        disable=not self.verbose,
                                        **tqdm_style,
                                        )

        results = pd.DataFrame()
        for pipeline_name in self.pipeline_names:
            results_pipeline = self._grid_test_pipeline(
                                                X=X,
                                                pipeline_name=pipeline_name,
                                                )
            results = pd.concat([results, results_pipeline])

        self.tqdm_progress.close()

        results = results[[
                            'pipeline', 
                            'split_number', 
                            'metric', 
                            'value', 
                            'splitter', 
                            'params', 
                            'train_id', 
                            'param_updates',
                            ]]
        
        return results.reset_index(drop=True)