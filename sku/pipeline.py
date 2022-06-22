import copy
import typing
import numpy as np
from sklearn.pipeline import Pipeline

class PipelineDD(Pipeline):
    def score(self, 
                X:typing.Dict[str, np.ndarray], 
                y:typing.Union[str, np.ndarray], 
                sample_weight=None):
        '''
        Transform the data, and apply ```score``` with the final estimator.
        Call ```transform``` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        ```score``` method. Only valid if the final estimator implements ```score```.
        
        
        Arguments
        ---------

        - ```X```: ```typing.Dict[str, np.ndarray]```: 
            A dictionary containing the data.
            For example:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.

        - ```y``` : ```typing.Union[str, np.ndarray]```:
            Please either pass a string, which corresponds to the 
            key in ```X``` which contains the labels, or pass
            the labels themselves.
            Defaults to ```None```.

        - ```sample_weight``` : ```array-like```:
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.
            Defaults to ```None```.
        
        
        Returns
        ---------
        - ```score``` : ```float```
            Result of calling ```score``` on the final estimator.


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