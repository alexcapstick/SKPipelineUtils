import inspect
import typing
import functools
import sys
import inspect

def get_default_args(func:typing.Callable):
    '''
    https://stackoverflow.com/a/12627202
    
    Allows you to collect the default arguments 
    of a function. This is useful for setting params
    in sklearn wrappers.
    
    
    Arguments
    ---------
    
    - func: typing.Callable:
        Function to get parameter defaults for.
    
    
    
    Returns
    --------
    
    - out: _type_: 
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



def partialclass_pickleable(name, cls, *args, **kwds):
    # from https://stackoverflow.com/a/58039373/19451559
    new_cls = type(name, (cls,), {
        '__init__': functools.partialmethod(cls.__init__, *args, **kwds)
    })

    # The following is copied nearly ad verbatim from :code:`namedtuple's` source.
    """
    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    """
    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return new_cls



def _prepare_func_on_args(func_on, input_dict, return_input_keys=False):

    pos_args = []
    kw_args = []
    input_keys = []

    if type(func_on) == list:
        if type(func_on[0]) == str:
            func_on = [func_on]
    if type(func_on) == dict:
        func_on = [func_on]
    
    for args in func_on:
        if type(args) == list:
            pos_args.append([input_dict[arg] for arg in args])
            kw_args.append({})
            input_keys.append(args)
        
        elif type(args) == dict:
            kw_args.append({f_arg: input_dict[i_arg] for f_arg, i_arg in args.items()})
            pos_args.append([])
            input_keys.append([i_arg for f_arg, i_arg in args.items()])
    
    if return_input_keys:
        return pos_args, kw_args, input_keys
    else:
        return pos_args, kw_args


def hasarg(func:typing.Callable, name:str) -> bool:
    '''
    Checks if the function can take the 
    name as an argument
    
    
    Arguments
    ---------
    
    - func: typing.Callable: 
        Function
    
    - name: str: 
        Name to check in the args
    
    
    
    Returns
    --------
    
    - out: bool: 
        True or False
    
    
    '''
    signature = inspect.signature(func)
    arg_list = [k for k, v in signature.parameters.items()]
    return name in arg_list