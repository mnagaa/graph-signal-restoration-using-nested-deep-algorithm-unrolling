import numpy as np


def calc_rmse(X1: np.ndarray, X2: np.ndarray):
    """Root mean squared error"""
    return float(np.sqrt(((X1 - X2) ** 2).mean()))


def min_max_scaler(x: np.ndarray):
    """MinMaxScaler"""
    mx = np.max(x)
    mi = np.min(x)
    return (x - mi) / (mx - mi)
