import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits
import typing
import copy


def train_test_group_split(
    *arrays,
    y,
    group,
    test_size: float = None,
    train_size: float = None,
    random_state: typing.Union[None, int] = None,
    shuffle: bool = True,
):
    """
    This function returns the train and test data given the
    split and the data. A single :code:`group` will not be in
    both the training and testing set. You should use either
    :code:`test_size` or :code:`train_size` but not both.



    Example
    ---------
    :code:`
    >>> (X_train, X_test,
        y_train, y_test,
        ids_train, ids_test) = train_test_group_split(X, y=y, group=group, test_size=0.33)

    :code:`


    Arguments
    ---------

    - arrays: array-like, optional:
        The data to split into training and testing sets. The labels and
        the group should be passed to :code:`y` and :code:`group` respectively.

    - y: array-like, optional:
        Label data with shape :code:`(n_samples)`,
        where :code:`n_samples` is the number of samples. These are the
        labels that are used to group the data into either the training
        or testing set.

    - group: array-like, optional:
        Event data with shape :code:`(n_samples)`,
        where :code:`n_samples` is the number of samples. These are the
        group ids that are used to group the data into either the training
        or testing set.

    - test_size: float, optional:
        This dictates the size of the outputted test set. This
        should be used if :code:`train_size=None`. If no :code:`test_size`
        or :code:`train_size` are given, then :code:`test_size` will default
        to :code:`0.25`
        Defaults to :code:`None`.

    - train_size: float, optional:
        This dictates the size of the outputted train set. This
        should be used if :code:`test_size=None`.
        Defaults to :code:`None`.

    - shuffle: bool, optional:
        dictates whether the data should be shuffled before the split
        is made.

    - random_state: None` or :code:`int, optional:
        This dictates the random seed that is used in the random
        operations for this function.



    Returns
    ----------

    - split arrays: list:
        This is a list of the input data, split into the training and
        testing sets. See the Example for an understanding of the
        order of the outputted arrays.


    """

    # setting defaults for test_size
    assert ~(test_size != None and train_size != None), (
        "Please supply " "either a train_size or a test_size"
    )
    assert len(arrays) > 0, "Please pass arrays to be split."

    if test_size is None:
        if not train_size is None:
            test_size = 1 - train_size
        else:
            test_size = 0.25

    # using the k fold splitter above
    splitter = StratifiedGroupKFold(
        n_splits=int(1 / test_size), shuffle=shuffle, random_state=random_state
    )

    splits = splitter.split(arrays[0], y=y, groups=group)

    # getting the split
    train_idx, test_idx = next(splits)

    # creating the output list
    output = []
    for array in arrays:
        output.append(array[train_idx])
        output.append(array[test_idx])

    output.append(y[train_idx])
    output.append(y[test_idx])
    output.append(group[train_idx])
    output.append(group[test_idx])

    return output


class DataPreSplit:
    def __init__(
        self,
        data: typing.Union[
            typing.List[
                typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]]
            ],
            typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]],
        ],
        split_fit_on: typing.List[str] = ["X", "y"],
    ):
        """
        This function allows you to wrap pre-split
        data into a class that behaves like
        an sklearn splitter. This is useful in
        pipeline searches.



        Examples
        ---------
        .. code-block::

            >>> splitter = sku.DataPreSplit(
                    data=[
                            (
                                {'X': np.arange(10)},
                                {'X': np.arange(5)},
                                ),
                            (
                                {'X': np.arange(5)},
                                {'X': np.arange(2)},
                                ),
                            (
                                {'X': np.arange(2)},
                                {'X': np.arange(3)},
                                ),
                        ],
                    split_fit_on=['X']
                    )
            >>> X = splitter.reformat_X()
            >>> for train_idx, val_idx in splitter.split(X['X']):
                    train_data, val_data = X['X'][train_idx], X['X'][val_idx]
                    do_things(train_data, val_data)


        Arguments
        ---------

        - data: typing.Union[ typing.List[ typing.Tuple[ typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]]], typing.Tuple[ typing.Dict[str, np.ndarray], typing.Dict[str, np.ndarray]]]:
            The pre-split data. Please ensure
            all splits have the same keys.

        - split_fit_on: typing.List[str], optional:
            The labels in the data dictionaries to split
            the data on.
            Defaults to :code:`['X', 'y']`.


        Raises
        ---------

            :code:`TypeError: If :code:`data` is not a list or tuple.


        """
        data_train, data_val = [], []

        if type(data) == tuple:
            data_train.append(data[0])
            data_val.append(data[1])
            self.n_splits = 1

        elif type(data) == list:
            for ns, (dtr, dte) in enumerate(data):
                data_train.append(dtr)
                data_val.append(dte)

            self.n_splits = ns + 1

        else:
            raise TypeError(
                "Please pass (data_train, data_val) "
                "as the argument to X, as a tuple. Alternatively, pass "
                "a list of tuples[(data_train, data_val), (data_train, data_val), ...]. "
            )

        self.data_train = copy.deepcopy(data_train)
        self.data_val = copy.deepcopy(data_val)
        self.split_fit_on = split_fit_on
        return

    def reformat_X(self) -> typing.Dict[str, np.ndarray]:
        """
        This reformats the X so that it can
        be split by the indices returned in :code:`splits`.
        It essentially concatenates all of the data
        dictionaries.


        Returns
        --------

        - out: typing.Dict[str, np.ndarray]:
            Dictionary containing concatenated
            arrays from the pre-split data.

        """

        self.train_idx, self.val_idx = [], []
        X = {}
        previous_end = 0
        for ns, (dtr, dte) in enumerate(zip(self.data_train, self.data_val)):
            X_ns = {}
            train_idx_ns = [dtr[k].shape[0] for k in self.split_fit_on]
            val_idx_ns = [dte[k].shape[0] for k in self.split_fit_on]
            for key, value in dte.items():
                if key in dtr:
                    X_ns[key] = np.concatenate([dtr[key], value], axis=0)

            if len(np.unique(train_idx_ns)) > 1:
                raise TypeError(
                    "Please ensure that all split_fit_on values in data_train "
                    "have the same length."
                )
            if len(np.unique(val_idx_ns)) > 1:
                raise TypeError(
                    "Please ensure that all split_fit_on values in data_val "
                    "have the same length."
                )

            val_idx_ns = val_idx_ns[0]
            train_idx_ns = train_idx_ns[0]

            self.val_idx.append(
                np.arange(train_idx_ns, train_idx_ns + val_idx_ns) + previous_end
            )
            self.train_idx.append(np.arange(train_idx_ns) + previous_end)
            previous_end += train_idx_ns + val_idx_ns

            for key, value in X_ns.items():
                if key in X:
                    X[key] = np.concatenate([X[key], value], axis=0)
                else:
                    X[key] = value

        return X

    def get_n_splits(self, groups: typing.Any = None) -> int:
        """
        Returns the number of splits.

        Arguments
        ---------

        - groups: typing.Any, optional:
            Ignored.
            Defaults to :code:`None`.



        Returns
        --------

        - out: int:
            The number of splits.


        """
        return self.n_splits

    def split(
        self,
        X: typing.Dict[str, np.ndarray],
        y: typing.Any = None,
        groups: typing.Any = None,
    ):
        """
        This returns the training and testing idices.



        Arguments
        ---------

        - X: typing.Dict[str, np.ndarray]:
            A data dictionary that is only
            used to ensure that an array of the
            right shape is being used for the splitting
            operation.

        - y: typing.Any, optional:
            Ignored.
            Defaults to :code:`None`.

        - groups: typing.Any, optional:
            Ignored.
            Defaults to :code:`None`.


        Returns
        --------

        - out:
            The train and test idices, wraped in a generator.
            See the Examples for an understanding of the output.


        """
        if X.shape[0] - 1 != self.val_idx[-1][-1]:
            raise TypeError(
                "X is not the same size as the indices built in " "reformat_X."
            )

        return zip(self.train_idx, self.val_idx)


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    """Repeated Stratified Group K-Fold cross validator.
    Repeats Stratified Group K-Fold n times with different randomization in each
    repetition.
    Read more in the :ref:`User Guide <repeated_k_fold>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    >>> rskf.get_n_splits(X, y)
    4
    >>> print(rskf)
    RepeatedStratifiedKFold(n_repeats=2, n_splits=2, random_state=36851234)
    >>> for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    ...
    Fold 0:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 1:
      Train: index=[0 3]
      Test:  index=[1 2]
    Fold 2:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 3:
      Train: index=[0 2]
      Test:  index=[1 3]
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    See Also
    --------
    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
