from joblib import Parallel


# package wide styling for progress bars
tqdm_style = {
                #'ascii':" ▖▘▝▗▚▞▉", 
                'ascii':"▏▎▍▋▊▉",
                'dynamic_ncols': True,
                }


class ProgressParallel(Parallel):
    def __init__(self, tqdm_bar, *args, **kwargs):
        self.tqdm_bar = tqdm_bar
        self.previously_completed = 0
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        difference = self.n_completed_tasks - self.previously_completed
        self.tqdm_bar.update(difference)
        self.tqdm_bar.refresh()
        self.previously_completed += difference