import numpy as np
import operator
from pie_toolbox.core.validators import check_data


def threshold(data: np.ndarray, threshold: float,
              op: operator = operator.gt, absolute: bool = False):
    """
    Threshold data by a given value.

    Parameters
    ----------
    data : array_like
        Data to be thresholded.
    threshold : float
        Threshold value.
    op : operator function, optional
        Operator to use for thresholding. Default is operator.gt (>).
    absolute : bool, optional
        If True, threshold the absolute values of the data. Default is False.

    Returns
    -------
    thresholded_data : array_like
        Thresholded data.
    """
    if absolute:
        data_abs = np.abs(data)
    else:
        data_abs = data
    return np.where(op(data_abs, threshold), data, 0)


def threshold_relative(data: np.ndarray, threshold_percentage: float,
                       op: operator = operator.gt, absolute: bool = False):
    """
    Threshold data by a given relative value.

    Parameters
    ----------
    data : array_like
        Data to be thresholded.
    threshold_percentage : float
        Relative threshold value (0-1).
    op : operator function, optional
        Operator to use for thresholding. Default is operator.gt (>).
    absolute : bool, optional
        If True, threshold the absolute values of the data. Default is False.

    Returns
    -------
    thresholded_data : array_like
        Thresholded data.
    """
    check_data.check_range(threshold_percentage, 0, 1)
    if absolute:
        max_value = np.max(np.abs(data))
    else:
        max_value = np.max(data)
    threshold_value = max_value * threshold_percentage
    return threshold(data, threshold_value, op, absolute)
