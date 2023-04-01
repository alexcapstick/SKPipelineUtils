import copy
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
from .pipeline import pipeline_constructor
from .progress import tqdm_style, ProgressParallel


def _get_relevant_param_updates(pipeline_name, pipeline_update_params):
    relevant_param_updates = {
        k: v
        for k, v in pipeline_update_params.items()
        if k.split("__")[0] in pipeline_name.split("--")
    }
    return relevant_param_updates


class PipelineBasicSearchCV(BaseEstimator):
    def __init__(
        self,
        pipeline_names: typing.List[str],
        name_to_object: typing.Dict[str, BaseEstimator],
        metrics: typing.Dict[str, typing.Callable],
        metrics_probability: typing.Dict[str, typing.Callable] = {},
        cv=None,
        repeat: int = 1,
        split_fit_on: typing.List[str] = ["X", "y"],
        split_transform_on: typing.List[str] = ["X", "y"],
        verbose: bool = False,
        n_jobs: int = 1,
        combine_splits: bool = False,
        combine_runs: bool = False,
        opt_metric: typing.Union[str, None] = None,
        minimise: bool = True,
    ):
        """
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines. You may also pass your own splits.

        Careful not to set a random_state in the objects passed in the
        :code:`name_to_object` dictionary, since each model gets cloned
        each time it is used in a pipeline, and the random_state will
        be the same in every run, and every split. Future code will
        allow the passing of a random state to this object directly.


        Example
        ---------

        .. code-block::

            name_to_object = {
                'gbt': sku.SKModelWrapperDD(
                    HistGradientBoostingClassifier,
                    fit_on=['X', 'y'],
                    predict_on=['X'],
                    ),
                'standard_scaler': sku.SKTransformerWrapperDD(
                    StandardScaler,
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

            pscv = PipelineBasicSearchCV(
                pipeline_names=pipeline_names,
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
            pscv.fit(X_data)
            results = pscv.cv_results_


        Arguments
        ---------

        - pipeline_names: typing.List[str]:
            This is a list of strings that describe the pipelines
            An example would be :code:`standard_scaler--ae--mlp`.
            The objects, separated by :code:'--' should be keys in
            :code:`name_to_object`.

        - name_to_object: typing.Dict[str, BaseEstimator]:
            A dictionary mapping the keys in :code:`pipeline_names` to
            the objects that will be used as transformers and
            models in the pipeline.

        - metrics: typing.Dict[str, typing.Callable]:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            :code:`func(labels, predictions)`.

        - metrics_probability: typing.Dict[str, typing.Callable]:
            A dictionary mapping the metric names to their callable
            functions. These functions should take the form:
            :code:`func(labels, prediction_probabilities)`.
            Defaults to :code:`{}`.

        - cv: sklearn splitting class, optional:
            This is the class that is used to produce the cross
            validation data. It should have the method
            :code:`.split(X, y, event)`, which returns the indices
            of the training and testing set, and the method
            :code:`get_n_splits()`, which should return the number
            of splits that this splitter was indended to make.
            Alternatively, if you pass :code:`None`,
            you may pass the training and validation data
            dictionaries themselves, in a tuple or list of tuples
            in the structure :code:`(data_train, data_val)`
            to the :code:`fit` method in place of the argument :code:`X`.
            Defaults to :code:`None`.

        - repeat: int, optional:
            The number of times to repeat the
            experiment.
            Defaults to :code:`1`.

        - split_fit_on: typing.List[str], optional:
            The keys corresponding to the values in
            the data dictionary passed in :code:`.fit()`
            that the :code:`cv` will take as positional
            arguments to the :code:`split()` function.
            If :code:`cv=None` then this is ignored.
            Defaults to :code:`['X', 'y']`.

        - split_transform_on: typing.List[str], optional:
            The keys corresponding to the values in
            the data dictionary passed in :code:`.fit()`
            that the :code:`cv` will split into training
            and testing data. This allows you to
            split data that isn't used in finding the
            splitting indices. If :code:`cv=None` then
            this is ignored.
            Defaults to :code:`['X', 'y']`.

        - verbose: bool, optional:
            Whether to print progress as the models are being tested.
            Remeber that you might also need to change the verbose, options
            in each of the objects given in :code:`name_to_object`.
            Defaults to :code:`False`.

        - n_jobs: int, optional:
            The number of parallel jobs. :code:`-1` will run the
            searches on all cores, but will incur significant memory
            and cpu cost.
            Defaults to :code:`1`.

        - combine_splits: bool, optional:
            Whether to combine the predictions
            over the splits before calculating
            the metrics. This can help reduce the variance
            in results when using Leave-One-Out.
            Defaults to :code:`False`.

        - combine_runs: bool, optional:
            Whether to combine the predictions
            over the runs before calculating
            the metrics. If :code:`True`,
            :code:`combine_splits` must also be :code:`True`.
            Defaults to :code:`False`.

        - opt_metric: typing.Union[str, None], optional:
            The metric values to use when determining the
            optimal parameters. If :code:`None`, the first
            metric given in :code:`metrics.keys()` will be used.
            If a :code:`str`, this should be a key in :code:`metrics`.
            Defaults to :code:`None`.

        - minimise: bool, optional:
            Whether to minimise the metric given in :code:`opt_metric`.
            If :code:`False`, the metric will be maximised.
            Defaults to :code:`True`.

        """

        if combine_runs:
            if not combine_splits:
                raise TypeError("If combine_runs=True, combine_splits must be True")

        self.pipeline_names = (
            pipeline_names if type(pipeline_names) == list else [pipeline_names]
        )
        self.name_to_object = name_to_object
        self.metrics = metrics
        self.metrics_probability = metrics_probability
        self.cv = cv
        self.repeat = repeat
        self.verbose = verbose
        self.split_fit_on = split_fit_on
        self.split_transform_on = split_transform_on
        self.split_runs = len(pipeline_names)
        self.n_jobs = n_jobs
        self.combine_splits = combine_splits
        self.combine_runs = combine_runs
        self.opt_metric = (
            opt_metric if not opt_metric is None else list(self.metrics.keys())[0]
        )
        self.minimise = 1 if minimise else -1

        return

    def _combine_runs_splits(
        self,
        results_single_split,
    ):

        labels_train, predictions_train, probabilities_train = (
            {nr: [] for nr in range(self.repeat)} for i in range(3)
        )
        labels_test, predictions_test, probabilities_test = (
            {nr: [] for nr in range(self.repeat)} for i in range(3)
        )
        for rss in results_single_split:
            ns, nr = rss[2], rss[3]
            labels_train[nr].append(rss[0][0])
            predictions_train[nr].append(rss[0][1])
            probabilities_train[nr].append(rss[0][2])

            labels_test[nr].append(rss[1][0])
            predictions_test[nr].append(rss[1][1])
            probabilities_test[nr].append(rss[1][2])

        for nr in range(self.repeat):

            labels_train[nr] = np.concatenate(labels_train[nr], axis=0)
            predictions_train[nr] = np.concatenate(predictions_train[nr], axis=0)
            if len(self.metrics_probability) > 0:
                probabilities_train[nr] = np.concatenate(
                    probabilities_train[nr], axis=0
                )
            labels_test[nr] = np.concatenate(labels_test[nr], axis=0)
            predictions_test[nr] = np.concatenate(predictions_test[nr], axis=0)
            if len(self.metrics_probability) > 0:
                probabilities_test[nr] = np.concatenate(probabilities_test[nr], axis=0)

        if self.combine_runs:

            labels_train = np.concatenate(list(labels_train.values()), axis=0)
            predictions_train = np.concatenate(list(predictions_train.values()), axis=0)
            if len(self.metrics_probability) > 0:
                probabilities_train = np.concatenate(
                    list(probabilities_train.values()), axis=0
                )
            labels_test = np.concatenate(list(labels_test.values()), axis=0)
            predictions_test = np.concatenate(list(predictions_test.values()), axis=0)
            if len(self.metrics_probability) > 0:
                probabilities_test = np.concatenate(
                    list(probabilities_test.values()), axis=0
                )

            # metrics
            results_single_split = [
                {
                    "metric": metric,
                    "value": func(info["labels"], info["predictions"]),
                    "repeat_number": np.nan,
                    "split_number": np.nan,
                    "train_or_test": info["tort"],
                }
                for metric, func in self.metrics.items()
                for info in [
                    {
                        "labels": labels_test,
                        "predictions": predictions_test,
                        "tort": "test",
                    },
                    {
                        "labels": labels_train,
                        "predictions": predictions_train,
                        "tort": "train",
                    },
                ]
            ]

            if len(self.metrics_probability) > 0:

                results_single_split.extend(
                    [
                        {
                            "metric": metric,
                            "value": func(info["labels"], info["probabilities"]),
                            "repeat_number": np.nan,
                            "split_number": np.nan,
                            "train_or_test": info["tort"],
                        }
                        for metric, func in self.metrics_probability.items()
                        for info in [
                            {
                                "labels": labels_test,
                                "probabilities": probabilities_test,
                                "tort": "test",
                            },
                            {
                                "labels": labels_train,
                                "probabilities": probabilities_train,
                                "tort": "train",
                            },
                        ]
                    ]
                )

        else:

            results_single_split = [
                {
                    "metric": metric,
                    "value": func(info["labels"], info["predictions"]),
                    "repeat_number": nr,
                    "split_number": np.nan,
                    "train_or_test": info["tort"],
                }
                for metric, func in self.metrics.items()
                for nr in range(self.repeat)
                for info in [
                    {
                        "labels": labels_test[nr],
                        "predictions": predictions_test[nr],
                        "tort": "test",
                    },
                    {
                        "labels": labels_train[nr],
                        "predictions": predictions_train[nr],
                        "tort": "train",
                    },
                ]
            ]

            if len(self.metrics_probability) > 0:

                results_single_split.extend(
                    [
                        {
                            "metric": metric,
                            "value": func(info["labels"], info["probabilities"]),
                            "repeat_number": nr,
                            "split_number": np.nan,
                            "train_or_test": info["tort"],
                        }
                        for metric, func in self.metrics_probability.items()
                        for nr in range(self.repeat)
                        for info in [
                            {
                                "labels": labels_test[nr],
                                "probabilities": probabilities_test[nr],
                                "tort": "test",
                            },
                            {
                                "labels": labels_train[nr],
                                "probabilities": probabilities_train[nr],
                                "tort": "train",
                            },
                        ]
                    ]
                )

        return [results_single_split]

    def _test_pipeline(self, X, y, pipeline):
        """
        Testing the whole pipeline, over the splits, with given params.
        """

        # defining testing function to run in parallel
        def _test_pipeline_parallel(
            train_data,
            test_data,
            y,
            ns,
            nr,
            pipeline,
            metrics,
            metrics_probability,
            combine_splits,
        ):

            pipeline.fit(train_data)
            predictions_train, out_data_train = pipeline.predict(
                train_data, return_data_dict=True
            )
            labels_train = out_data_train[y]

            predictions_test, out_data_test = pipeline.predict(
                test_data, return_data_dict=True
            )
            labels_test = out_data_test[y]

            if len(metrics_probability) > 0:
                probabilities_train = pipeline.predict_proba(train_data)
                probabilities_test = pipeline.predict_proba(test_data)
            else:
                probabilities_train = None
                probabilities_test = None

            if combine_splits:
                return [
                    [labels_train, predictions_train, probabilities_train],
                    [labels_test, predictions_test, probabilities_test],
                    ns,
                    nr,
                ]

            # metrics
            results_single_split = [
                {
                    "metric": metric,
                    "value": func(info["labels"], info["predictions"]),
                    "repeat_number": nr,
                    "split_number": ns,
                    "train_or_test": info["tort"],
                }
                for metric, func in metrics.items()
                for info in [
                    {
                        "labels": labels_test,
                        "predictions": predictions_test,
                        "tort": "test",
                    },
                    {
                        "labels": labels_train,
                        "predictions": predictions_train,
                        "tort": "train",
                    },
                ]
            ]

            # probability metrics
            results_single_split.extend(
                [
                    {
                        "metric": metric,
                        "value": func(info["labels"], info["probabilities"]),
                        "repeat_number": nr,
                        "split_number": ns,
                        "train_or_test": info["tort"],
                    }
                    for metric, func in metrics_probability.items()
                    for info in [
                        {
                            "labels": labels_test,
                            "probabilities": probabilities_test,
                            "tort": "test",
                        },
                        {
                            "labels": labels_train,
                            "probabilities": probabilities_train,
                            "tort": "train",
                        },
                    ]
                ]
            )

            return results_single_split

        # parallel running of fitting
        f_parallel = functools.partial(
            _test_pipeline_parallel,
            y=y,
            metrics=self.metrics,
            metrics_probability=self.metrics_probability,
            combine_splits=self.combine_splits,
        )
        try:
            results_single_split = ProgressParallel(
                tqdm_bar=self.tqdm_progress, n_jobs=self.n_jobs, backend="threading"
            )(
                joblib.delayed(f_parallel)(
                    pipeline=copy.deepcopy(pipeline),
                    train_data=train_data,
                    test_data=test_data,
                    ns=ns,
                    nr=nr,
                )
                for ns, (train_data, test_data) in enumerate(X)
                for nr in range(self.repeat)
            )
            kbi = False
        except KeyboardInterrupt:
            kbi = True

        # delete parallel processes
        get_reusable_executor().shutdown(wait=True)

        if kbi:
            raise KeyboardInterrupt

        if self.combine_splits:
            results_single_split = self._combine_runs_splits(results_single_split)

        return results_single_split

    def _param_test_pipeline(self, X, y, pipeline_name):
        """
        Testing the whole pipeline, over the splits. This
        should be overwritten when using different param grid
        functions.
        """

        self.tqdm_progress.set_postfix({"pm_n": pipeline_name.split("--")[-1]})

        results_pipeline = pd.DataFrame()

        pipeline = pipeline_constructor(
            pipeline_name, name_to_object=self.name_to_object, verbose=False
        )

        results_temp = {
            "pipeline": pipeline_name,
            "metrics": [],
            "splitter": type(self.cv).__name__,
            "params": pipeline.get_params(),
            "param_updates": None,
            "train_id": uuid.uuid4(),
        }

        # getting and saving results
        results_temp_metrics = self._test_pipeline(
            X=X,
            y=y,
            pipeline=pipeline,
        )

        for rtm in results_temp_metrics:
            results_temp["metrics"].extend(rtm)

        results_pipeline = pd.concat(
            [
                results_pipeline,
                pd.json_normalize(
                    results_temp,
                    record_path="metrics",
                    meta=[
                        "pipeline",
                        "splitter",
                        "params",
                        "train_id",
                        "param_updates",
                    ],
                ),
            ]
        )

        return results_pipeline

    def fit(
        self,
        X: typing.Union[
            typing.Dict[str, np.ndarray],
            typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]],
            typing.List[
                typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]]
            ],
        ],
        y: str,
    ) -> pd.DataFrame:
        """
        This function fits and predicts the pipelines, 
        with the, optional parameters and splitting 
        arguments and produces a table of results
        given the metrics.

        Arguments
        ---------

        - X: typing.Union[\
                typing.Dict[str, np.ndarray], \
                typing.Tuple[\
                    typing.Dict[str, np.ndarray], \
                    typing.Dict[str, np.ndarray]\
                    ],\
                typing.List[\
                    typing.Tuple[\
                        typing.Dict[str, np.ndarray], \
                        typing.Dict[str, np.ndarray]\
                        ]\
                    ],\
                ]:
            The data dictionary that will be used to run
            the experiments. This may also be a tuple of 
            data dictionaries used for training and validation
            splits if :code:`cv=None`. In this case, you may also pass
            a list of tuples of data dictionaries in the form
            :code:`[(data_train, data_val), (data_train, data_val), ...]`, 
            where :code:`data_train` and :code:`data_val` are data dictionaries.
        
        - y: str:
            Please pass a string, which corresponds to the 
            key in :code:`X` which contains the labels.
        
        
        
        """

        # check if train and val are pre split
        if self.cv is None:
            if type(X) not in [tuple, list]:
                raise TypeError(
                    "If using cv=None, please pass (data_train, data_val) "
                    "as the argument to X, as a tuple. Alternatively, pass "
                    "a list of tuples[(data_train, data_val), (data_train, data_val), ...]. "
                )
            if type(X) == tuple:
                X = [X]
            self.n_splits = len(X)
        else:
            assert type(X) == dict, "If cv is not None, ensure that X is a dictionary."
            train_test_idx = list(
                self.cv.split(*[X[split_data] for split_data in self.split_fit_on])
            )
            self.n_splits = self.cv.get_n_splits(groups=X[self.split_fit_on[-1]])
            X = [
                (
                    {
                        split_data: X[split_data][train_idx]
                        for split_data in self.split_transform_on
                    },
                    {
                        split_data: X[split_data][test_idx]
                        for split_data in self.split_transform_on
                    },
                )
                for train_idx, test_idx in train_test_idx
            ]

        self.tqdm_progress = tqdm.tqdm(
            total=(self.split_runs * self.n_splits * self.repeat),
            desc="Searching",
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

        results = results[
            [
                "pipeline",
                "repeat_number",
                "split_number",
                "train_or_test",
                "metric",
                "value",
                "splitter",
                "params",
                "train_id",
                "param_updates",
            ]
        ]

        self.cv_results_ = results.reset_index(drop=True)

        return

    @property
    def best_params_(self):

        opt_metric = self.opt_metric
        minimise = self.minimise

        best_train_ids = (
            self.cv_results_.query("metric == @opt_metric & train_or_test == 'test'")[
                ["pipeline", "value", "train_id"]
            ]
            .groupby(["pipeline", "train_id"])
            .mean()
            .sort_values(by="value", ascending=bool(minimise + 1))
            .reset_index()
            .drop_duplicates(subset="pipeline")["train_id"]
            .values
        )

        best_model_results = (
            self.cv_results_.query("train_id in @best_train_ids")
            .drop_duplicates(subset="pipeline")[["pipeline", "param_updates"]]
            .reset_index(drop=True)
            .set_index("pipeline")
            .to_dict()["param_updates"]
        )

        return best_model_results


class PipelineSearchCV(PipelineBasicSearchCV):
    def __init__(
        self,
        *args,
        param_grid: typing.Union[
            typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]
        ] = None,
        **kwargs,
    ):
        """
        This is the same as PipelineBasicSearchCV except
        a parameter grid can also be passed, allowing
        the user to test multiple configurations of each pipeline.

        Example
        ---------

        .. code-block::

            pscv = PipelineSearchCV(
                pipeline_names=pipeline_names,
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
            pscv.fit(X_data)
            results = pscv.cv_results_



        Arguments
        ---------

        - args:
            All the same args as PipelineSearchCV.

        - param_grid: typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]:
            A dictionary or list of dictionaries that
            are used as a parameter grid for testing performance
            of pipelines with multiple hyper-parameters. This should
            be of the usual format given to
            :code:`sklearn.model_selection.ParameterGrid` when
            used with :code:`sklearn.pipeline.Pipeline`.
            If :code:`None`, then the pipeline is tested with
            the parameters given to the objects in
            :code:`name_to_object`.
            All pipelines will be tested with the given parameters
            in addition to the parameter grid passed.
            Only those parameters relevant to a pipeline
            will be used.
            Defaults to :code:`None`.

        - kwargs:
            All the same kwargs as PipelineSearchCV.

        """

        super().__init__(*args, **kwargs)

        # building a param grid and counting number of experiments
        self.param_grid = {
            pipeline_name: [None] for pipeline_name in self.pipeline_names
        }
        if not param_grid is None:
            self.split_runs = 0
            # all combinations of parameters (including duplicates)
            param_grid_all = list(ParameterGrid(param_grid))

            # de-duplicating and saving only runs relevant for each pipeline
            for pipeline_name in self.pipeline_names:
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
                self.param_grid[pipeline_name].extend(
                    list(map(dict, param_grid_pipeline_name))
                )
                # counting number of runs for that pipeline
                self.split_runs += len(self.param_grid[pipeline_name])
        else:
            self.split_runs = len(self.pipeline_names)

        return

    def _param_test_pipeline(
        self,
        X,
        y,
        pipeline_name,
    ):
        """
        Testing the whole pipeline, over the splits and params.
        """

        self.tqdm_progress.set_postfix({"pm_n": pipeline_name.split("--")[-1]})

        results_pipeline = pd.DataFrame()

        for g in self.param_grid[pipeline_name]:
            pipeline = pipeline_constructor(
                pipeline_name,
                name_to_object=self.name_to_object,
                verbose=False,
            )
            results_temp = {
                "pipeline": pipeline_name,
                "metrics": [],
                "splitter": type(self.cv).__name__,
                "params": pipeline.get_params(),
                "param_updates": g,
                "train_id": uuid.uuid4(),
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
                results_temp["metrics"].extend(rtm)

            results_pipeline = pd.concat(
                [
                    results_pipeline,
                    pd.json_normalize(
                        results_temp,
                        record_path="metrics",
                        meta=[
                            "pipeline",
                            "splitter",
                            "params",
                            "train_id",
                            "param_updates",
                        ],
                    ),
                ]
            )

        return results_pipeline


class PipelineBayesSearchCV(PipelineBasicSearchCV):
    def __init__(
        self,
        *args,
        param_grid: typing.List[typing.Dict[str, typing.List[typing.Any]]] = None,
        max_iter: int = 10,
        **kwargs,
    ):
        """
        This class allows you to test multiple pipelines
        on a supervised task, reporting on the metrics given in
        a table of results.
        Given a splitting function, it will perform cross validation
        on these pipelines. A parameter grid can also be passed, allowing
        the user to test multiple configurations of each pipeline.
        This class allows you to perform a bayesian parameter search over
        the parameters given in :code:`param_grid`.
        Note: At the moment this only supports real value param searches.

        Example
        ---------

        .. code-block::

            pscv = PipelineBayesSearchCV(
                pipeline_names=pipeline_names,
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
            pscv.fit(X_data)
            results = pscv.cv_results_


        Arguments
        ---------

        - param_grid: typing.List[typing.Dict[str, typing.List[typing.Any]]]:
            A dictionary or list of dictionaries that
            are used as a parameter grid for testing performance
            of pipelines with multiple hyper-parameters. This should
            be of the usual format given to
            :code:`sklearn.model_selection.ParameterGrid` when
            used with :code:`sklearn.pipeline.Pipeline`.
            If :code:`None`, then the pipeline is tested with
            the parameters given to the objects in
            :code:`name_to_object`.
            All pipelines will be tested with the given parameters
            in addition to the parameter grid passed.
            Only those parameters relevant to a pipeline
            will be used.
            Defaults to :code:`None`.

        - max_iter: int, optional:
            The number of calls to make on each pipeline when
            finding the, optimum params.
            Defaults to :code:`10`.

        """

        super().__init__(
            *args,
            **kwargs,
        )

        # building a param grid and counting number of experiments
        self.param_grid = {pipeline_name: [] for pipeline_name in self.pipeline_names}
        if not param_grid is None:
            self.split_runs = 0
            # all combinations of parameters (including duplicates)
            param_grid_all = list(param_grid)

            for pipeline_name in self.pipeline_names:
                # pipeline specific parameter grid
                for pipeline_update_params in param_grid_all:
                    pipeline_update_params = _get_relevant_param_updates(
                        pipeline_name, pipeline_update_params
                    )
                    if len(pipeline_update_params) != 0:
                        self.param_grid[pipeline_name].append(pipeline_update_params)
                if len(self.param_grid[pipeline_name]) > 0:
                    self.split_runs += len(self.param_grid[pipeline_name]) * max_iter
                else:
                    self.split_runs += 1
        else:
            self.split_runs = len(self.pipeline_names)

        self.n_jobs = self.n_jobs
        self.max_iter = max_iter
        self.opt_result = {}

        return

    def _param_test_pipeline(
        self,
        X,
        y,
        pipeline_name,
    ):
        """
        Testing the whole pipeline, over the splits and params.
        """

        self.tqdm_progress.set_postfix({"pm_n": pipeline_name.split("--")[-1]})

        results_pipeline = []
        if len(self.param_grid[pipeline_name]) == 0:
            opt_params = []
        else:
            opt_params = [d for d in self.param_grid[pipeline_name][0].items()]
            opt_params = [
                {"name": items[0], "low": items[1][0], "high": items[1][1]}
                for items in opt_params
            ]
        dims = [
            skopt.space.Integer(**items)
            if (type(items["low"]) == int and type(items["high"]) == int)
            else skopt.space.Real(**items)
            for items in opt_params
        ]

        def to_optimise(**params_update):
            pipeline = pipeline_constructor(
                pipeline_name, name_to_object=self.name_to_object, verbose=False
            )
            results_temp = {
                "pipeline": pipeline_name,
                "metrics": [],
                "splitter": type(self.cv).__name__,
                "params": pipeline.get_params(),
                "param_updates": params_update,
                "train_id": uuid.uuid4(),
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
                results_temp["metrics"].extend(rtm)

            results_temp_df = pd.json_normalize(
                results_temp,
                record_path="metrics",
                meta=[
                    "pipeline",
                    "splitter",
                    "params",
                    "train_id",
                    "param_updates",
                ],
            )

            results_pipeline.append(
                results_temp_df,
            )
            metric_opt = self.opt_metric
            opt_result = results_temp_df.query(
                "train_or_test == 'test' " "& metric == @metric_opt"
            )["value"].mean()

            return opt_result * self.minimise

        if not pipeline_name in self.opt_result:
            self.opt_result[pipeline_name] = []
        if len(dims) > 0:

            @skopt.utils.use_named_args(dimensions=dims)
            def named_args_to_optimise(*args, **kwargs):
                return to_optimise(*args, **kwargs)

            self.opt_result[pipeline_name].append(
                skopt.gp_minimize(
                    named_args_to_optimise,
                    dimensions=dims,
                    n_calls=self.max_iter,
                )
            )
            self.opt_result[pipeline_name][-1].fun *= self.minimise
            self.opt_result[pipeline_name][-1].func_vals *= self.minimise
        else:
            to_optimise()
            self.opt_result[pipeline_name].append(None)

        return pd.concat(results_pipeline)
