import numpy as np
from scipy import interpolate


def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return minmax_custom(arr, arr_min, arr_max)


def minmax_custom(arr, min, max):
    return (arr - min) / (max - min)


def minmax_over_nonzero(arr):
    arr_min = np.min(arr[np.nonzero(arr)])
    arr_max = np.max(arr)
    return minmax_custom(arr, arr_min, arr_max)


def minmax_reverse(arr, min, max):
    return arr * (max - min) + min


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging over nonzero elements."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    arr2 = arr.reshape(shape)
    cond = (arr2 > 0).sum(axis=(1, 3))
    out = np.zeros(new_shape)
    np.true_divide(arr2.sum(axis=(1, 3)), cond, where=(cond) > 0, out=out)
    return out


def interpolate_on_missing(arr, equal_to=0, method='nearest'):
    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])
    # mask invalid values
    array = np.ma.masked_equal(arr, equal_to)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    res = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method=method, fill_value=0)
    if np.min(res) < 0:
        mask = (res >= 0).astype(int)
        res = res * mask
    return res
