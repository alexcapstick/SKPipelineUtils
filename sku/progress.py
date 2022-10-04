from joblib import Parallel
import aml

ProgressParallel = aml.ProgressParallel

# package wide styling for progress bars
tqdm_style = {
                #'ascii':" ▖▘▝▗▚▞▉", 
                'ascii':"▏▎▍▋▊▉",
                'dynamic_ncols': True,
                }
