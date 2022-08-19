from abc import abstractmethod
import numpy as np
import pandas as pd
import typing
import tqdm
import uuid
import joblib
import functools
import skopt

from joblib.externals.loky import get_reusable_executor

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from .pipeline import PipelineDD, pipeline_constructor
from .progress import tqdm_style, ProgressParallel




def _get_relevant_param_updates(pipeline_name, pipeline_update_params):
    relevant_param_updates = {
        k:v 
        for k,v in pipeline_update_params.items() 
        if k.split('__')[0] in pipeline_name
        }
    return relevant_param_updates



###### For future: Maybe combine the LOO predictions!! Not calc metrics on separate splits



class PipelineBasicSearchCV(BaseEstimator):
    def __init__(self,
                    pipeline_names:typing.List[str],
                    name_to_object:typing.Dict[str, BaseEstimator],
                    metrics:typing.Dict[str, typing.Callable],
                    cv=None,
                    split_fit_on:typing.List[str]=['X', 'y'],
                    split_transform_on:typing.List[str]=['X', 'y'],
                    verbose:bool=False,
                    n_jobs=1,
                    ):
        '''
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in 
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines.

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

        pscv = PipelineBasicSearchCV(pipeline_names=pipeline_names,
                                    name_to_object=name_to_object,
                                    metrics=metrics,
                                    cv=splitter,
                                    split_fit_on=['X', 'y'],
                                    split_transform_on=['X', 'y', 'id'],
                                    verbose=True,
                                    )
        X_data = {
                    'X': X_labelled, 'y': y_labelled, 'id': id_labelled,
                    'X_unlabelled': X_unlabelled, 'id_unlabelled': id_unlabelled,
                    }
        results = pscv.fit(X_data)
        ```

        Arguments
        ---------

        - `pipeline_names`: `typing.List[str]`:
            This is a list of strings that describe the pipelines
            An example would be `standard_scaler--ae--mlp`.
            The objects, separated by '--' should be keys in 
            `name_to_object`.
        
        - `name_to_object`: `typing.Dict[str, BaseEstimator]`:
            A dictionary mapping the keys in `pipeline_names` to
            the objects that will be used as transformers and 
            models in the pipeline.
        
        - `metrics`: `typing.Dict[str, typing.Callable]`:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            `func(labels, predictions)`.
        
        - `cv`: sklearn splitting class:
            This is the class that is used to produce the cross
            validation data. It should have the method
            `.split(X, y, event)`, which returns the indices
            of the training and testing set, and the method 
            `get_n_splits()`, which should return the number
            of splits that this splitter was indended to make.
            Defaults to `None`.

        - `split_fit_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will take as positional
            arguments to the `split()` function.
            Defaults to `['X', 'y']`.        

        - `split_transform_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will split into training 
            and testing data. This allows you to 
            split data that isn't used in finding the
            splitting indices.
            Defaults to `['X', 'y']`.               
        
        - `verbose`: `bool`:
            Whether to print progress as the models are being tested.
            Remeber that you might also need to change the verbose options
            in each of the objects given in `name_to_object`.
            Defaults to `False`.     
        
        - `n_jobs`: `int`:
            The number of parallel jobs. `-1` will run the 
            searches on all cores, but will incur significant memory 
            and cpu cost.
            Defaults to `1`.     
        
        '''
        assert not cv is None, 'Currently cv=None is not supported. '\
                                'Please pass an initialised sklearn splitter.'

        self.pipeline_names = pipeline_names
        self.name_to_object = name_to_object
        self.metrics = metrics
        self.cv = cv
        self.verbose = verbose
        self.split_fit_on = split_fit_on
        self.split_transform_on = split_transform_on

        self.split_runs = len(pipeline_names)
        self.n_jobs = n_jobs

        return

    def _test_pipeline(self,
                    X,
                    y,
                    pipeline,
                    ):
        '''
        Testing the whole pipeline, over the splits, with given params.
        '''

        # defining testing function to run in parallel
        def _test_pipeline_parallel(
                                    X,
                                    y,
                                    train_idx,
                                    test_idx,
                                    ns,
                                    split_transform_on,
                                    pipeline,
                                    metrics,
                                    ):
            # data to split on
            train_data = { split_data:X[split_data][train_idx] for split_data in split_transform_on }
            test_data = { split_data:X[split_data][test_idx] for split_data in split_transform_on }
            # other data
            train_data.update({ k: v for k, v in X.items() if k not in train_data })
            test_data.update({ k: v for k, v in X.items() if k not in test_data })
            
            pipeline.fit(train_data)
            predictions_train, out_data_train = pipeline.predict(train_data, return_data_dict=True)
            labels_train = out_data_train[y]

            predictions_test, out_data_test = pipeline.predict(test_data, return_data_dict=True)
            labels_test = out_data_test[y]

            results_single_split = [
                                    {
                                    'metric': metric, 
                                    'value': func(labels_test, predictions_test),
                                    'split_number': ns,
                                    'train_or_test': 'test',
                                    #'train_positve': np.sum(train_y_out)/train_y_out.shape[0],
                                    #'test_positve': np.sum(labels)/labels.shape[0],
                                    } 
                                    for metric, func in metrics.items()
                                    ]

            results_single_split.extend([
                                        {
                                        'metric': metric, 
                                        'value': func(labels_train, predictions_train),
                                        'split_number': ns,
                                        'train_or_test': 'train',
                                        #'train_positve': np.sum(train_y_out)/train_y_out.shape[0],
                                        #'test_positve': np.sum(labels)/labels.shape[0],
                                        } 
                                        for metric, func in metrics.items()
                                        ])
            


            return results_single_split
        
        # parallel running of fitting
        f_parallel = functools.partial(_test_pipeline_parallel, 
                                X=X, 
                                y=y,
                                split_transform_on=self.split_transform_on,
                                pipeline=pipeline,
                                metrics=self.metrics,
        )
        try:
            results_single_split = ProgressParallel(
                                    tqdm_bar=self.tqdm_progress, 
                                    n_jobs=self.n_jobs
                                    )(joblib.delayed(f_parallel)(
                                        train_idx=train_idx,
                                        test_idx=test_idx,
                                        ns=ns,
                                        ) for ns, (train_idx, test_idx) 
                                            in enumerate(
                                                list(
                                                    self.cv.split(*[ X[split_data] 
                                                        for split_data in self.split_fit_on 
                                                    ]))))
            kbi = False

        except KeyboardInterrupt:
            kbi = True
        
        # delete parallel processes
        get_reusable_executor().shutdown(wait=True)

        if kbi:
            raise KeyboardInterrupt

        return results_single_split

    def _param_test_pipeline(self,
                            X,
                            y,
                            pipeline_name,
                            ):
        '''
        Testing the whole pipeline, over the splits. This
        should be overwritten when using param grids.
        '''

        results_pipeline = pd.DataFrame()

        pipeline = pipeline_constructor(pipeline_name,
                    name_to_object=self.name_to_object,
                    verbose=False)
        results_temp = {
                        'pipeline': pipeline_name,
                        'metrics': [],
                        'splitter': type(self.cv).__name__,
                        'params': pipeline.get_params(),
                        'param_updates': None,
                        'train_id': uuid.uuid4(),
                        }
        
        # getting and saving results
        results_temp_metrics = self._test_pipeline(
                                            X=X,
                                            y=y,
                                            pipeline=pipeline,
                                            )
        for rtm in results_temp_metrics:
            results_temp['metrics'].extend(rtm)

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
            y:str, 
            ) -> pd.DataFrame:
        '''
        This function fits and predicts the pipelines, 
        with the optional parameters and splitting 
        arguments and produces a table of results
        given the metrics.

        Arguments
        ---------

        - `X`: `typing.Dict[str, np.ndarray]`:
            The data dictionary that will be used to run
            the experiments.
        
        - `y` : `str`:
            Please either pass a string, which corresponds to the 
            key in `X` which contains the labels.
        
        Returns
        ---------
        - `results`: `pandas.DataFrame`:
            The results, with columns:
            `['pipeline', 'split_number', 'metric', 
            'value', 'splitter', 'params', 'train_id', 
            'param_updates']`
        
        
        
        '''

        self.tqdm_progress = tqdm.tqdm(
                                        total=(self.split_runs
                                                *self.cv.get_n_splits(groups=X[self.split_fit_on[-1]])
                                                ), 
                                        desc='Searching', 
                                        disable=not self.verbose,
                                        **tqdm_style,
                                        )

        results = pd.DataFrame()
        for pipeline_name in self.pipeline_names:
            results_pipeline = self._param_test_pipeline(
                                                X=X,
                                                y=y,
                                                pipeline_name=pipeline_name,
                                                )
            results = pd.concat([results, results_pipeline])
        self.tqdm_progress.close()

        results = results[[
                            'pipeline', 
                            'split_number', 
                            'train_or_test',
                            'metric', 
                            'value', 
                            'splitter', 
                            'params', 
                            'train_id', 
                            'param_updates',
                            ]]
        
        return results.reset_index(drop=True)





























class PipelineSearchCV(PipelineBasicSearchCV):
    def __init__(self,
                    pipeline_names:typing.List[str],
                    name_to_object:typing.Dict[str, BaseEstimator],
                    metrics:typing.Dict[str, typing.Callable],
                    cv=None,
                    split_fit_on:typing.List[str]=['X', 'y'],
                    split_transform_on:typing.List[str]=['X', 'y'],
                    param_grid:typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]=None,
                    verbose:bool=False,
                    n_jobs=1,
                    ):
        '''
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in 
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines. A parameter grid can also be passed, allowing
        the user to test multiple configurations of each pipeline.
        This class allows you to perform a grid search over the parameters
        given in `param_grid`.

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

        - `pipeline_names`: `typing.List[str]`:
            This is a list of strings that describe the pipelines
            An example would be `standard_scaler--ae--mlp`.
            The objects, separated by '--' should be keys in 
            `name_to_object`.
        
        - `name_to_object`: `typing.Dict[str, BaseEstimator]`:
            A dictionary mapping the keys in `pipeline_names` to
            the objects that will be used as transformers and 
            models in the pipeline.
        
        - `metrics`: `typing.Dict[str, typing.Callable]`:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            `func(labels, predictions)`.
        
        - `cv`: sklearn splitting class:
            This is the class that is used to produce the cross
            validation data. It should have the method
            `.split(X, y, event)`, which returns the indices
            of the training and testing set, and the method 
            `get_n_splits()`, which should return the number
            of splits that this splitter was indended to make.
            Defaults to `None`.

        - `split_fit_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will take as positional
            arguments to the `split()` function.
            Defaults to `['X', 'y']`.        

        - `split_transform_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will split into training 
            and testing data. This allows you to 
            split data that isn't used in finding the
            splitting indices.
            Defaults to `['X', 'y']`.        

        - `param_grid`: `typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]`:
            A dictionary or list of dictionaries that 
            are used as a parameter grid for testing performance
            of pipelines with multiple hyper-parameters. This should
            be of the usual format given to 
            `sklearn.model_selection.ParameterGrid` when 
            used with `sklearn.pipeline.Pipeline`.
            If `None`, then the pipeline is tested with 
            the parameters given to the objects in 
            `name_to_object`.
            All pipelines will be tested with the given parameters
            in addition to the parameter grid passed.
            Only those parameters relevant to a pipeline
            will be used.
            Defaults to `None`.        
        
        - `verbose`: `bool`:
            Whether to print progress as the models are being tested.
            Remeber that you might also need to change the verbose options
            in each of the objects given in `name_to_object`.
            Defaults to `False`.     
        
        - `n_jobs`: `int`:
            The number of parallel jobs. `-1` will run the 
            searches on all cores, but will incur significant memory 
            and cpu cost.
            Defaults to `1`.     
        
        '''
        assert not cv is None, 'Currently cv=None is not supported. '\
                                'Please pass an initialised sklearn splitter.'

        super().__init__(
            pipeline_names = pipeline_names,
            name_to_object = name_to_object,
            metrics = metrics,
            cv = cv,
            verbose = verbose,
            split_fit_on = split_fit_on,
            split_transform_on = split_transform_on,
            )

        # building a param grid and counting number of experiments
        self.param_grid = {pipeline_name: [None] for pipeline_name in pipeline_names}
        if not param_grid is None:
            self.split_runs = 0
            # all combinations of parameters (including duplicates)
            param_grid_all = list(ParameterGrid(param_grid))
            
            # de-duplicating and saving only runs relevant for each pipeline
            for pipeline_name in pipeline_names:
                # pipeline specific parameter grid
                param_grid_pipeline_name = []
                for pipeline_update_params in param_grid_all:
                    # building a list of unique dictionaries
                    param_grid_pipeline_name.append(
                        frozenset(
                            _get_relevant_param_updates(
                                pipeline_name=pipeline_name,
                                pipeline_update_params=pipeline_update_params,
                                ).items()
                            )
                        )
                param_grid_pipeline_name = frozenset(param_grid_pipeline_name)
                self.param_grid[pipeline_name].extend(list(map(dict, param_grid_pipeline_name)))
                # counting number of runs for that pipeline
                self.split_runs += len(self.param_grid[pipeline_name])
        else:
            self.split_runs = len(pipeline_names)
        self.n_jobs = n_jobs

        return


    def _param_test_pipeline(self,
                            X,
                            y,
                            pipeline_name,
                            ):
        '''
        Testing the whole pipeline, over the splits and params.
        '''

        results_pipeline = pd.DataFrame()

        for g in self.param_grid[pipeline_name]:
            pipeline = pipeline_constructor(pipeline_name,
                        name_to_object=self.name_to_object,
                        verbose=False)
            results_temp = {
                            'pipeline': pipeline_name,
                            'metrics': [],
                            'splitter': type(self.cv).__name__,
                            'params': pipeline.get_params(),
                            'param_updates': g,
                            'train_id': uuid.uuid4(),
                            }
            # updating params if there are any to update
            if not g is None:
                pipeline.set_params(**g)
            
            # getting and saving results
            results_temp_metrics = self._test_pipeline(
                                                X=X,
                                                y=y,
                                                pipeline=pipeline,
                                                )
            for rtm in results_temp_metrics:
                results_temp['metrics'].extend(rtm)
            
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
















class PipelineBayesSearchCV(PipelineBasicSearchCV):
    def __init__(self,
                    pipeline_names:typing.List[str],
                    name_to_object:typing.Dict[str, BaseEstimator],
                    metrics:typing.Dict[str, typing.Callable],
                    cv=None,
                    split_fit_on:typing.List[str]=['X', 'y'],
                    split_transform_on:typing.List[str]=['X', 'y'],
                    param_grid:typing.List[typing.Dict[str, typing.List[typing.Any]]]=None,
                    max_iter:int=10,
                    opt_metric:typing.Union[str, None]=None,
                    minimise:bool=True,
                    verbose:bool=False,
                    n_jobs:int=1,
                    ):
        '''
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in 
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines. A parameter grid can also be passed, allowing
        the user to test multiple configurations of each pipeline.
        This class allows you to perform a bayesian parameter search over 
        the parameters given in `param_grid`.
        Note: At the moment this only supports real value param searches.

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

        pscv = PipelineBayesSearchCV(pipeline_names=pipeline_names,
                                    name_to_object=name_to_object,
                                    metrics=metrics,
                                    cv=splitter,
                                    split_fit_on=['X', 'y'],
                                    split_transform_on=['X', 'y', 'id'],
                                    verbose=True,
                                    param_grid= [
                                        {'flatten_gbt__learning_rate': [5, 20]},
                                        {'flatten_standard_scaler__with_mean': [True, False]},
                                        {'flatten_mlp__dropout': [0.2, 0.9]},
                                        ],
                                    )
        X_data = {
                    'X': X_labelled, 'y': y_labelled, 'id': id_labelled,
                    'X_unlabelled': X_unlabelled, 'id_unlabelled': id_unlabelled,
                    }
        results = pscv.fit(X_data)
        ```


        
        Arguments
        ---------

        - `pipeline_names`: `typing.List[str]`:
            This is a list of strings that describe the pipelines
            An example would be `standard_scaler--ae--mlp`.
            The objects, separated by '--' should be keys in 
            `name_to_object`.
        
        - `name_to_object`: `typing.Dict[str, BaseEstimator]`:
            A dictionary mapping the keys in `pipeline_names` to
            the objects that will be used as transformers and 
            models in the pipeline.
        
        - `metrics`: `typing.Dict[str, typing.Callable]`:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            `func(labels, predictions)`.
        
        - `cv`: sklearn splitting class:
            This is the class that is used to produce the cross
            validation data. It should have the method
            `.split(X, y, event)`, which returns the indices
            of the training and testing set, and the method 
            `get_n_splits()`, which should return the number
            of splits that this splitter was indended to make.
            Defaults to `None`.

        - `split_fit_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will take as positional
            arguments to the `split()` function.
            Defaults to `['X', 'y']`.        

        - `split_transform_on`: `typing.List[str]`:
            The keys corresponding to the values in 
            the data dictionary passed in `.fit()`
            that the `cv` will split into training 
            and testing data. This allows you to 
            split data that isn't used in finding the
            splitting indices.
            Defaults to `['X', 'y']`.        

        - `param_grid`: `typing.List[typing.Dict[str, typing.List[typing.Any]]]`:
            A dictionary or list of dictionaries that 
            are used as a parameter grid for testing performance
            of pipelines with multiple hyper-parameters. This should
            be of the usual format given to 
            `sklearn.model_selection.ParameterGrid` when 
            used with `sklearn.pipeline.Pipeline`.
            If `None`, then the pipeline is tested with 
            the parameters given to the objects in 
            `name_to_object`.
            All pipelines will be tested with the given parameters
            in addition to the parameter grid passed.
            Only those parameters relevant to a pipeline
            will be used.
            Defaults to `None`.  

        - `max_iter`: `int`, optional:
            The number of calls to make on each pipeline when
            finding the optimum params.
            Defaults to `10`.

        - `opt_metric`: `typing.Union[str, None]`, optional:
            The metric values to use when determining the 
            optimal parameters. If `None`, the first
            metric given in `metrics.keys()` will be used.
            If a `str`, this should be a key in `metrics`.
            Defaults to `None`.
        
        - `minimise`: `bool`, optional:
            Whether to minimise the metric given in `opt_metric`.
            If `False`, the metric will be maximised.
            Defaults to `True`.
        
        - `verbose`: `bool`:
            Whether to print progress as the models are being tested.
            Remeber that you might also need to change the verbose options
            in each of the objects given in `name_to_object`.
            Defaults to `False`.     
        
        - `n_jobs`: `int`:
            The number of parallel jobs. `-1` will run the 
            searches on all cores, but will incur significant memory 
            and cpu cost.
            Defaults to `1`.     
        
        '''
        assert not cv is None, 'Currently cv=None is not supported. '\
                                'Please pass an initialised sklearn splitter.'

        super().__init__(
            pipeline_names = pipeline_names,
            name_to_object = name_to_object,
            metrics = metrics,
            cv = cv,
            verbose = verbose,
            split_fit_on = split_fit_on,
            split_transform_on = split_transform_on,
            )

        # building a param grid and counting number of experiments
        self.param_grid = {pipeline_name: [] for pipeline_name in pipeline_names}
        if not param_grid is None:
            self.split_runs = 0
            # all combinations of parameters (including duplicates)
            param_grid_all = list(param_grid)
            
            for pipeline_name in pipeline_names:
                # pipeline specific parameter grid
                for pipeline_update_params in param_grid_all:
                    pipeline_update_params = _get_relevant_param_updates(
                                                pipeline_name, 
                                                pipeline_update_params
                                                )
                    if len(pipeline_update_params) != 0:
                        self.param_grid[pipeline_name].append(pipeline_update_params)
                self.split_runs += len(self.param_grid[pipeline_name])
        else:
            self.split_runs = len(pipeline_names)
        self.split_runs *= max_iter
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.opt_metric = opt_metric if not opt_metric is None else list(self.metrics.keys())[0]
        self.opt_result = {}
        self.minimise = 1 if minimise else -1


        return


    def _param_test_pipeline(self,
                            X,
                            y,
                            pipeline_name,
                            ):
        '''
        Testing the whole pipeline, over the splits and params.
        '''

        results_pipeline = []

        opt_params = [
            list(d.items())[0] 
            for d in self.param_grid[pipeline_name]
            ]
        opt_params = [
            {'name': items[0], 'low': items[1][0], 'high': items[1][1]} 
            for items in opt_params]
        dims = [skopt.space.Real(**items) for items in opt_params]
        
        @skopt.utils.use_named_args(dimensions=dims)
        def to_optimise(**params_update):
            pipeline = pipeline_constructor(pipeline_name,
                        name_to_object=self.name_to_object,
                        verbose=False)
            results_temp = {
                'pipeline': pipeline_name,
                'metrics': [],
                'splitter': type(self.cv).__name__,
                'params': pipeline.get_params(),
                'param_updates': params_update,
                'train_id': uuid.uuid4(),
                }
            # updating params if there are any to update
            if not params_update is None:
                pipeline.set_params(**params_update)
            # getting and saving results
            results_temp_metrics = self._test_pipeline(
                                                X=X,
                                                y=y,
                                                pipeline=pipeline,
                                                )
            for rtm in results_temp_metrics:
                results_temp['metrics'].extend(rtm)

            results_temp_df = pd.json_normalize(results_temp, 
                                record_path='metrics', 
                                meta=['pipeline', 
                                        'splitter', 
                                        'params', 
                                        'train_id',
                                        'param_updates',
                                        ])

            results_pipeline.append(
                                    results_temp_df,
                                    )
            metric_opt = self.opt_metric
            opt_result = (results_temp_df.query(
                "train_or_test == 'test' "\
                "& metric == @metric_opt")
                ['value']
                .mean()
                )

            return opt_result*self.minimise

        self.opt_result[pipeline_name] = skopt.gp_minimize(
            to_optimise, 
            dimensions=dims, 
            n_calls=self.max_iter,
            )
        
        self.opt_result[pipeline_name].fun *= self.minimise
        self.opt_result[pipeline_name].func_vals *= self.minimise

        results_pipeline = pd.concat(results_pipeline)

        return results_pipeline
