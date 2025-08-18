import numpy as np


def check_type(data, data_type):
    assert isinstance(
        data, data_type), f"Data is of type {type(data)} but should be of type {data_type}"


def check_dimensions_number(data, dimensions):
    check_type(data, np.ndarray)
    check_type(dimensions, int)
    assert data.ndim == dimensions, "Data has wrong number of dimensions"


def check_dimensions(data, dimensions):
    check_type(data, np.ndarray)
    check_type(dimensions, tuple)
    check_dimensions_number(data, len(dimensions))
    for i, dim in enumerate(dimensions):
        assert data.shape[i] == dim, f"Dimension {i} is {data.shape[i]} but should be {dim}"


def check_not_none_or_empty(data):
    """Check if data is not None or empty"""
    assert data is not None, "Data is None"
    if isinstance(data, str):
        assert data != "", "Data is empty"
    if isinstance(data, np.ndarray):
        assert np.prod(data.shape) != 0, "Data is empty"


def check_equal_axes(data: tuple, axes: tuple):
    """Check if dimensions of specific axes are equal"""
    if isinstance(axes, int):
        axes = [axes]
    assert all(data[0].shape[ax] == data[1].shape[ax]
               for ax in axes), f"Dimensions {axes} are not equal across data"


def check_range(data: int | float | np.ndarray,
                min_value: int | float, max_value: int | float):
    """
    Check if data is within the given range.

    Parameters
    ----------
    data : int or float or numpy.ndarray
        Data to be checked
    min_value : int or float
        Minimum value of the range
    max_value : int or float
        Maximum value of the range

    Raises
    ------
    AssertionError
        If data is below min_value or above max_value
    """
    data_min = data if isinstance(data, (int, float)) else np.min(data)
    data_max = data if isinstance(data, (int, float)) else np.max(data)
    assert data_min >= min_value, f"Data is below minimum value {min_value}"
    assert data_max <= max_value, f"Data is above maximum value {max_value}"
