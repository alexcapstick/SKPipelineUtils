import copy
from sklearn.pipeline import Pipeline

class PipelineDD(Pipeline):
    def score(self, X, y=None, sample_weight=None):
        '''
        Transform the data, and apply ```score``` with the final estimator.
        Call ```transform``` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        ```score``` method. Only valid if the final estimator implements ```score```.
        
        
        Arguments
        ---------

        - ```X```: ```typing.Dict[str, typing.Union[np.ndarray, typing.Dict[str, np.ndarray]]]```: 
            A dictionary containing the data.
            For example, if ```semi_supervised=True```:
            ```
            X = {
                    'labelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    'unlabelled': {'X': X_DATA, 'y': Y_DATA, **kwargs},
                    }
            ```
            If ```semi_supervised=False```:
            ```
            X = {'X': X_DATA, 'y': Y_DATA, **kwargs}
            ```.

        - ```y``` : iterable, default=None
            Ignored. Please pass targets to X, which should be a dictionary.
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
        
        if 'labelled' in X_copy:
            y_pass = X_copy['labelled']['y']
        
        if 'X' in X_copy:
            y_pass = X_copy['y']
        
        return self.steps[-1][1].score(X_copy, y_pass, **score_params)