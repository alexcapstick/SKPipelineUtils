from joblib import Parallel
from aml.utils.parallel import ProgressParallel

# package wide styling for progress bars
tqdm_style = {
                #'ascii':" ▖▘▝▗▚▞▉", 
                'ascii':"▏▎▍▋▊▉",
                'dynamic_ncols': True,
                }
