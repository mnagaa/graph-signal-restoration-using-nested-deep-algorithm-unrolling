import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix


def vectorize(arr):
    return arr.T.reshape(-1)

def inv_vectorize(vector, shape):
    return np.reshape(vector, shape, order='F')

def commutation_matrix(A):
    """
    Cite:
    https://stackoverflow.com/questions/60678746/compute-commutation-matrix-in-numpy-scipy-efficiently
    """
    m, n = A.shape[0], A.shape[1]
    row = np.arange(m*n)
    col = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K

def clip(x: scipy.sparse.csr_matrix, _min=None, _max=None):
    """
    See np.clip function.
    """
    if _min is None:
        raise ValueError('_min argument is needed.')
    if _max is None:
        raise ValueError('_max argument is needed.')
    if _min > _max:
        raise ValueError(f'_min and _max should satisfy _min <= _max, but _min={_min}, _max={_max}.')

    y = np.where(
        x.data > _max,
        _max,  # _maxを超えているとき->_max
        # _maxを超えていない時
        np.where(
            x.data < _min,
            _min,  # _minをより小さい時->_min
            x.data)
        )
    return csr_matrix(y.reshape(x.shape), shape=x.shape)
