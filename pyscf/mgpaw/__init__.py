# pyscf/isdfx/__init__.py

import os
import ctypes
from pyscf import lib
import numpy

# Load the shared library
libevalao = lib.load_library('libevalao')

# Bind functions
ndpointer = numpy.ctypeslib.ndpointer

getNimg = libevalao.getNimg
getNimg.restype = ctypes.c_double
getNimg.argtypes = [ctypes.c_double, ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
]

formImages = libevalao.formImages
formImages.restype = None
formImages.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                       ctypes.c_int,
                       ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                       ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

eval_all_aos = libevalao.eval_all_aos
eval_all_aos.restype = None
eval_all_aos.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ctypes.c_int,ctypes.c_int,ctypes.c_int,
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         ctypes.c_int,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

pbceval_all_aos = libevalao.pbceval_all_aos
pbceval_all_aos.restype = None
pbceval_all_aos.argtypes = [ctypes.c_int, ndpointer(numpy.complex128, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ctypes.c_int,ctypes.c_int,ctypes.c_int,
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),ctypes.c_int,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ctypes.c_int,
                         ndpointer(numpy.complex128, flags="C_CONTIGUOUS")]

