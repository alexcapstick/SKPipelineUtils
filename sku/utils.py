import inspect
import typing

def get_default_args(func:typing.Callable):
    '''
    https://stackoverflow.com/a/12627202
    
    Allows you to collect the default arguments 
    of a function. This is useful for setting params
    in sklearn wrappers.
    
    
    Arguments
    ---------
    
    - ```func```: ```typing.Callable```:
        Function to get parameter defaults for.
    
    
    
    Returns
    --------
    
    - ```out```: ```_type_``` : 
        _description_
    
    
    '''
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }