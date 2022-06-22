import inspect
import typing
import functools


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




def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
        
    NewCls.__name__ = cls.__name__
    return NewCls