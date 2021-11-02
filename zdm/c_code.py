""" Codes related to C """
import os
import ctypes
from pkg_resources import resource_filename
from scipy import LowLevelCallable

lib_path = os.path.join(
    resource_filename('zdm', 'src'), 'zdmlib.so')

lib = ctypes.CDLL(os.path.abspath(lib_path))
lib.lognormal_dlog_c.restype = ctypes.c_double
lib.lognormal_dlog_c.argtypes = (ctypes.c_int, 
                  ctypes.POINTER(ctypes.c_double))

func_ll = LowLevelCallable(lib.lognormal_dlog_c)