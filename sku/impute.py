import typing
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.impute import KNNImputer
from .utils import get_default_args
import tqdm
from .progress import tqdm_style, ProgressParallel
import joblib
import copy

class KDTreeKNNImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, k:int=10, verbose:bool=True, n_jobs:int=1, **kwargs):
        '''
        This imputer allows you to impute a dataset
        based on a background dataset with a set
        of numeric groups, in which the transformed
        data will be measured against before doing 
        the imputation. This is best understood with
        an example.       
        
        Any of the arguments passed to the class, apart from :code:`k`
        control the behaviour of the KNN Imputation done on the 
        new data using the background data 
        (see :code:`sklearn.impute.KNNImputer`).

        
        Examples
        ---------
        
        Here, we try to impute the data of new people, based
        on a background data set from other people. We would like 
        to impute the data of our new people using only the data
        from the other people who have the 5 closest ages.::
            
            >>> imputer = KDTreeKNNImputer(k=5) # k=5 -> the 5 closest ages
            >>> # builds a tree with the background data and their ages
            >>> imputer.fit(background_data, kdtree_values=background_ages)
            >>> # impute the new data, using the new ages to filter the background data
            >>> imputer.transform(new_data, kdtree_values=new_data_ages)
        
        
        Arguments
        ---------
        
        - k: int, optional:
            The number of neighbours to use when filtering the data. 
            Defaults to :code:`10`.
        
        - verbose: bool, optional:
            Print verbose.
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel jobs.
            Defaults to :code:`1`.

        - **kwargs:
            The keyword arguments that will be passed to KNNImputer
        

        Attributes
        ----------

        - fit_all: list:
            The values in :code:`kdtree_values` that were transformed using
            all of the data. This happens when there is a 
            column in the filtered background data that has all 
            missing values.
        
        - kdtree: scipy.spatial.KDTree:
            The tree fit to the :code:`kdtree_values`.
        
        '''
        self._params = get_default_args(KNNImputer)
        self._params.update(**kwargs)
        self.knn_arg_names = list(get_default_args(KNNImputer).keys())
        self.kdtree = None
        self._params['k'] = k
        self.fit_all = set([])
        self.verbose = verbose
        self._params['n_jobs'] = n_jobs

    @classmethod
    def _get_param_names(self):
        '''Get parameter names for the estimator'''
        params = ['k', 'n_jobs']
        params.extend(list(get_default_args(KNNImputer).keys()))
        return sorted(params)

    def get_params(self=None, deep=True) -> dict:
        if not self is None:
            return self._params
        else:
            return KDTreeKNNImputer()._params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self._get_param_names():
                self._params[key] = value
            else:
                raise TypeError(f"{key} is not an acceptable parameter.")
        return super().set_params(**params)

    def fit(self, X:np.ndarray, y:typing.Any=None, kdtree_values:typing.Union[None, np.ndarray]=None):
        '''
        Fit the tree to the background data.       
        
        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The background data to use for the imputation.
        
        - y: typing.Any, optional:
            Ignored. 
            Defaults to :code:`None`.
        
        - kdtree_values: typing.Union[None, np.ndarray], optional:
            The background values to use in the filtering. See the example.
            Currently, only a an array of shape (n,) and (1, n) is accepted.
            Defaults to :code:`None`.
        
        
        '''
        assert not kdtree_values is None, "Please make sure to pass kdtree_values."
        assert len(kdtree_values) == len(X), "Ensure that kdtree_values and X are the same length."

        self.background_X = X
        kdtree_values = np.copy(kdtree_values)
        if len(kdtree_values.shape) == 1:
            kdtree_values = kdtree_values.reshape(-1, 1)

        na_idx = np.any(pd.isna(kdtree_values), axis=-1)
        # removing the NAs from the values
        kdtree_values = kdtree_values[~na_idx]
        self.background_X = self.background_X[~na_idx]
        self.kdtree = scipy.spatial.KDTree(data=kdtree_values)
        return self

    def transform(
        self, X:np.ndarray, kdtree_values:typing.Union[None, np.ndarray]=None
        )->np.ndarray:
        '''
        Transform the new data using the background
        data, using the kdtree_values.
        
        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The new X values to perform the imputation on.
        
        - kdtree_values: typing.Union[None, np.ndarray], optional:
            The values to use in the filtering of the background data.
            Currently, only a an array of shape (n,) and (n,1) is accepted.
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - out: np.ndarray: 
            The imputed array.
        
        
        '''
        assert not kdtree_values is None, "Please make sure to pass kdtree_values."
        assert len(kdtree_values) == len(X), "Ensure that kdtree_values and X are the same length."

        # copying input array
        X = np.copy(X)

        if len(kdtree_values.shape) > 1:
            kdtree_values = kdtree_values.reshape(-1,)

        na_idx = pd.isna(kdtree_values)

        pbar = tqdm.tqdm(
            desc='Imputing',
            total=len(np.unique(kdtree_values[~na_idx])) + int(np.sum(na_idx) > 1),
            disable=not self.verbose,
            **tqdm_style
            )

        if np.sum(na_idx) > 1:
            knn_imputer = KNNImputer(
                **{key: self._params[key] for key in self.knn_arg_names}
                )
            knn_imputer.fit(X[na_idx])
            X[na_idx] = knn_imputer.transform(X[na_idx])
            kdtree_values = kdtree_values[~na_idx]
            pbar.update(1)
            pbar.refresh()
        # finding unique values to do the imputation on
        kdtree_values_unique = np.unique(kdtree_values)

        X_impute = X[~na_idx]

        def _impute(kdt_value, imputer, k, kdtree, background_data):
            _, i = kdtree.query(kdt_value, k=k)
            fit_data = (
                background_data[i] if len(background_data[i].shape)>1 
                else background_data[i].reshape(1,-1)
                )
            if np.any(np.all(pd.isna(fit_data), axis=0)):
                fit_data = background_data
                fit_all = True
            else:
                fit_all = False
            imputer.fit(fit_data)
            X_impute[kdtree_values == kdt_value] = imputer.transform(X_impute[kdtree_values == kdt_value])

            return fit_all

        imputer = KNNImputer(
                **{key: self._params[key] for key in self.knn_arg_names}
                )

        results = ProgressParallel(
            tqdm_bar=pbar, backend="threading", n_jobs=self._params['n_jobs']
            )(
                joblib.delayed(_impute)(
                    kdt_v, copy.deepcopy(imputer), k=self._params['k'], 
                    kdtree=self.kdtree, background_data=self.background_X,
                    )
                for kdt_v in kdtree_values_unique
                )

        for result, kdt_v in zip(results, kdtree_values_unique):
            if result:
                self.fit_all.add(kdt_v)

        pbar.close()
        X[~na_idx] = X_impute

        return X

    def fit_transform(
        self, X:np.ndarray, y:typing.Any=None, kdtree_values:typing.Union[None, np.ndarray]=None
        )->np.ndarray:

        self.fit(X=X, y=y, kdtree_values=kdtree_values)
        return self.transform(X=X, kdtree_values=kdtree_values)