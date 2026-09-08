#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 9/4/24, 6:56 AM
#     Last change by yifei
#    *****************************************************************************

import subprocess

import numpy
import numpy as np
import scipy
import scipy.sparse.linalg as splinalg


# try:
#     import cupy as cp
#     import cupy.sparse.linalg as cpsplinalg
# except ImportError:
#     # logging.warning(f"CuPy is not installed or not available!")
#     print(f"CuPy is not installed or not available!")

def create_matrix_of_zeros(size, use_cuda=False, sparse_matrix=False):
    if use_cuda:
        if sparse_matrix:
            return cp.sparse.csr_matrix((size, size))
        else:
            return cp.zeros(shape=(size, size))
    else:
        if sparse_matrix:
            return scipy.sparse.csr_matrix((size, size))
        else:
            return numpy.zeros(shape=(size, size))


def list_to_array(data: list, use_cuda=False):
    """
    Convert a Python list into a numpy.array or cupy.array
    :param data: Python list of data
    :param use_cuda: boolean variable to indicate whether to use CUDA
    :return:
    """
    if use_cuda:
        return cp.array(data)
    else:
        return np.array(data)

def is_cuda_available():
    """
    Check if CUDA is available on the system by running 'nvidia-smi'.

    Returns:
    bool: True if CUDA is available, False otherwise.
    """
    try:
        # Run the 'nvidia-smi' command
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def solve_sparse_linear_system(A, b, use_sparse=False, use_cuda=False):
    """
    Solve the linear system Ax = b

    Parameters:
    A: matrix
    b: vector of known variables

    Returns:
    x: unknown variables
    """

    if use_cuda:
        if use_sparse:
            x = cpsplinalg.spsolve(A, b)
        else:
            x = cp.linalg.solve(A, b)
    else:
        if use_sparse:
            x = splinalg.spsolve(A, b)
        else:
            x = np.linalg.solve(A, b)
    return x

