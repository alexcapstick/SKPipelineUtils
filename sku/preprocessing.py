from __future__ import annotations

import sklearn
import logging
import typing
import copy
import warnings
import numpy as np

from sklearn.preprocessing import StandardScaler

import aml

flatten = aml.flatten

from .flatten_wrapper import FlattenWrapper
from .utils import partialclass_pickleable

class StandardGroupScaler(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        '''
        This class allows you to scale the data based on a group.

        When calling transform, if the group has not been seen 
        in the fitting method, then the global statistics will
        be used to scale the data (global = across all groups).

        Where the mean or standard deviation are equal to :code:`NaN`,
        in any axis on any group, that particular value will be 
        replaced with the global mean or standard deviation for that
        axis (global = across all groups). If the standard deviation 
        is returned as :code:`0.0` then the global standard deviation 
        and mean is used.
        
        '''
        self.scalers = {}
        self.means_ = {}
        self.vars_ = {}
        self.global_scalar = None
        self.global_mean_ = None
        self.global_var_ = None
        self.scalars_fitted = False
        self.groups_fitted = []
    
    def fit(self, 
                X:np.ndarray, 
                y:typing.Union[np.ndarray, None]=None, 
                groups:typing.Union[np.ndarray, None]=None,
                ) -> StandardGroupScaler:
        '''
        Compute the mean and std to be used for later scaling.
        
        
        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The data used to compute the mean and standard deviation used for later
            scaling along the features axis. This should be of shape 
            :code:`(n_samples, n_features)`.

        - y: typing.Union[np.ndarray, None], optional:
            Igorned. 
            Defaults to :code:`None`.

        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.
        

        
        
        
        Returns
        --------
        
        - self: sku.StandardGroupScaler:
            The fitted scaler.
        
        
        '''
        # if no groups are given then all points are
        # assumed to be from the same group
        if groups is None:
            logging.warning('You are using the grouped version of StandardScaler, yet you have '\
                            'not passed any groups. Using sklearn.preprocessing.StandardScaler '\
                            'will be faster if you have no groups to use.')
            groups = np.ones((X.shape[0]))
        
        self.global_mean_ = np.nanmean(X, axis=0)
        self.global_var_ = np.nanvar(X, axis=0)
        
        # creating an instance of the sklearn StandardScaler
        # for each group
        groups_unique = np.unique(groups)
        for group_name in groups_unique:
            # get the data from that group
            mask = groups == group_name
            X_sub = X[mask]

            # calculating the statistics
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Mean of empty slice.*')
                # calculating mean
                group_means = np.nanmean(X_sub, axis=0)
                warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.*')
                # calculating var
                group_vars = np.nanvar(X_sub, axis=0)
            
            # replace NaN with global statistics
            replace_with_global_mask = np.isnan(group_means) | np.isnan(group_vars) | (group_vars == 0)
            group_means[replace_with_global_mask] = self.global_mean_[replace_with_global_mask]
            group_vars[replace_with_global_mask] = self.global_var_[replace_with_global_mask]

            # saving group statistics
            self.means_[group_name] = group_means
            self.vars_[group_name] = group_vars

            self.groups_fitted.append(group_name)

        # flag to indicate the scalars have been fitted
        self.scalars_fitted = True
        
        return self
    
    def transform(self, 
                    X:np.ndarray, 
                    y:typing.Union[np.ndarray, None]=None,
                    groups:typing.Union[np.ndarray, None]=None, 
                    ) -> np.ndarray:
        '''
        Perform standardization by centering and scaling by group.

        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The data used to scale along the features axis. This should be of shape 
            :code:`(n_samples, n_features)`.

        - y: typing.Union[np.ndarray, None], optional:
            Ignored. 
            Defaults to :code:`None`.

        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.
        
        

        Returns
        --------
        
        - X_norm: np.ndarray: 
            The transformed version of :code:`X`.
        
        
        '''
        
        X_norm = copy.deepcopy(X)

        # if no groups are given then all points are
        # assumed to be from the same group
        if groups is None:
            logging.warning('You are using the grouped version of StandardScaler, yet you have '\
                            'not passed any groups. Using sklearn.preprocessing.StandardScaler '\
                            'will be faster if you have no groups to use.')
            groups = np.ones((X_norm.shape[0]))

        # transforming the data in each group
        groups_unique = np.unique(groups)
        for group_name in groups_unique:
            mask = groups == group_name
            try:
                X_norm[mask] = (X_norm[mask] - self.means_[group_name])/np.sqrt(self.vars_[group_name])
            except KeyError:
                X_norm[mask] = (X_norm[mask] - self.global_mean_)/np.sqrt(self.global_var_)
            
        return X_norm
    

    def fit_transform(self, 
                        X:np.ndarray, 
                        y:typing.Union[np.ndarray, None]=None,
                        groups:typing.Union[np.ndarray, None]=None, 
                        ) -> np.ndarray:
        '''
        Fit to data, then transform it. Fits transformer to X using the groups
        and returns a transformed version of X.
        
        
        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The data used to compute the mean and standard deviation used for later
            scaling along the features axis. This should be of shape 
            :code:`(n_samples, n_features)`.
        
        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.
        
        - y: typing.Union[np.ndarray, None], optional:
            Igorned. 
            Defaults to :code:`None`.
        
        
        
        Returns
        --------
        
        - self:
            The fitted scaler.
        
        
        '''

        self.fit(X=X, groups=groups, y=y)
        return self.transform(X=X, groups=groups, y=y)




FlattenStandardScaler = partialclass_pickleable(
                            name='FlattenWrapper',
                            cls=FlattenWrapper,
                            estimator=StandardScaler, 
                            unflatten_transform=True,
                            )

flatten_standard_scaler_doc = {
    '__init__': (
        '''
        This class allows you to scale the data to 0 mean and unit standard 
        deviation, based on statistics calculated over a flattened version 
        of the array. The :code:`start_dim` and :code:`end_dim` allow you to 
        choose where to flatten the array. By default, the flattening
        will allow for mean values to calculated over an array 
        of shape (in_shape[0], -1). Please see the flattening operations
        in :code:`aml.flatten` to understand the arguments 
        :code:`start_dim` and :code:`end_dim`.
        
        
        Examples
        ---------


        Arguments
        ---------
        
        - start_dim: int, optional:
            The first dim to flatten. 
            Defaults to :code:`0`.
        
        - end_dim: int, optional:
            The last dim to flatten. 
            Defaults to :code:`-1`.
        
        '''),

    'fit': (
        '''
        Compute the mean and std to be used for later scaling.


        Arguments
        ---------
        
        - X: np.ndarray: 
            The data used to compute the mean and standard deviation used for later
            scaling along the features axis. This should be of shape 
            :code:`(n_samples, n_features)`.

        - y: typing.Union[np.ndarray, None], optional:
            Igorned. 
            Defaults to :code:`None`.


        Returns
        --------
        
        - self: FlattenStandardScalerOld:
            The fitted scaler.
        
        
        '''),

    'transform': (
        '''
        Perform standardization on data.

        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The data used to scale along the features axis. This should be of shape 
            :code:`(n_samples, n_features)`.
       

        Returns
        --------
        
        - X_norm: np.ndarray: 
            The transformed version of :code:`X`.
        
        
        '''),
}

FlattenStandardScaler.__init__.__doc__ = flatten_standard_scaler_doc['__init__']
FlattenStandardScaler.fit.__doc__ = flatten_standard_scaler_doc['fit']
FlattenStandardScaler.transform.__doc__ = flatten_standard_scaler_doc['transform']












class Flatten(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(
        self,
        start_dim:int=0,
        end_dim:int=-1,
        copy=False,
        )->None:
        '''
        This class allows you to flatten an array inside a pipeline.
        This class was implemented to mirror the behaviour in 
        :code:`https://pytorch.org/docs/stable/generated/torch.flatten.html`.
        
        
        Examples
        ---------
        .. code-block:: 

            >>> flat = Flatten(start_dim=1, end_dim=-1)
            >>> flat.fit(None, None) # ignored
            >>> flat.transform(
                    [[[1, 2],
                    [3, 4]],
                    [[5, 6],
                    [7, 8]]]
                    )
            [[1,2,3,4],
            [5,6,7,8]]

        
        Arguments
        ---------
        
        - start_dim: int, optional:
            The first dim to flatten. 
            Defaults to :code:`0`.
        
        - end_dim: int, optional:
            The last dim to flatten. 
            Defaults to :code:`-1`.
        
        - copy: bool, optional:
            Whether to return a copied version
            of the array during the transform
            method.
            Defaults to :code:`False`.
        
        '''
        if end_dim != -1: 
            raise NotImplementedError('Currently only end_dim = -1 is supported')
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.copy = copy
        return
    
    def fit(
        self,
        X:np.ndarray=None, 
        y:None=None, 
        ) -> Flatten:
        '''
        This function is required for the pipelines to work, but is ignored.
        
        Arguments
        ---------
        
        - X: 
            Ignored.
            Defaults to :code:`None`.

        - y:
            Igorned. 
            Defaults to :code:`None`.
        

        Returns
        --------
        
        - self:
            The fitted scaler.
        
        
        '''
        return self
    
    def transform(
        self,
        X:np.ndarray,
        )->np.ndarray:
        '''
        This will transform the array by returning a flattened
        version.
        
        
        Arguments
        ---------
        
        - X: np.ndarray: 
            The array to be flattened.
        
        Returns
        --------
        
        - out: np.ndarray: 
            The flattened array.
        
        
        '''
        if self.copy:
            X_out = copy.deepcopy(X)
        else:
            X_out = X
        return flatten(
                X_out, 
                start_dim=self.start_dim, 
                end_dim=self.end_dim, 
                copy=self.copy,
                )



