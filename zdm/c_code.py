""" Codes related to C """
import os
import ctypes
import importlib.resources as resources
from scipy import LowLevelCallable

lib_path = os.path.join(
    resources.files('zdm').joinpath('src'), 'zdmlib.so')

if not os.path.isfile(lib_path):
    raise ImportError("You need to create zdmlib.so!!")

lib = ctypes.CDLL(os.path.abspath(lib_path))
lib.lognormal_dlog_c.restype = ctypes.c_double
lib.lognormal_dlog_c.argtypes = (ctypes.c_int, 
                  ctypes.POINTER(ctypes.c_double))

func_ll = LowLevelCallable(lib.lognormal_dlog_c)
