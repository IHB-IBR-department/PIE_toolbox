import numpy as np
from pie_toolbox.core.common.logger import setup_root_logger, get_logger
from pie_toolbox.core.ssm_pca.utils import subtract_mean
from pie_toolbox.core.validators import check_data
setup_root_logger('log_norm')
logger = get_logger('log_norm')


def log_voxels(matrix: np.ndarray) -> np.ndarray:
    """
    Sets all nonpositive values of a matrix to 0.0, and
    takes the natural logarithm of all positive values.

    Parameters
    ----------
    matrix : numpy.ndarray
        2D array of shape (n_subjects, n_voxels). Each row is a subject.

    Returns
    -------
    matrix : numpy.ndarray
        The transformed matrix.
    """
    matrix[(matrix <= 0)] = 0.0
    matrix[(matrix > 0)] = np.log(matrix[(matrix > 0)])
    return matrix


def normalize(matrix: np.ndarray, seed_mask: np.ndarray = None):
    """
    Normalizes the matrix by row (inside each subject) and then by column (between subjects).

    Parameters
    ----------
    matrix : numpy.ndarray
        2D array of shape (n_subjects, n_voxels). Each row is a subject.
    seed_mask : numpy.ndarray, optional
        Boolean mask array for the seed region. If not None, normalize by the mean of the seed region.

    Returns
    -------
    result : tuple
        A tuple of a normalized matrix and the global mean profile.
        matrix : numpy.ndarray
            The normalized matrix (n_subjects, n_voxels). Each row is a subject.
        gmp : numpy.ndarray
            The global mean profile.
    """

    logger.debug('Normalization inside each subject')
    matrix = np.apply_along_axis(subtract_mean, 1, matrix, seed_mask)
    logger.debug('Normalization between subjects')
    gmp = np.mean(matrix, axis=0)
    matrix = np.apply_along_axis(subtract_mean, 0, matrix)
    return matrix, gmp


def log_normalize(matrix: np.ndarray, seed_mask: np.ndarray = None,
                  log_transform: bool = True):
    """
    Applies logarithmic transformation to the matrix and normalizes it.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (n_subjects, n_voxels). Each row is a subject.
    seed_mask : np.ndarray, optional
        Boolean mask array for the seed region. If not None, normalize by the mean of the seed region.
    log_transform : bool, optional
        If True, applies logarithmic transformation to the matrix.

    Returns
    -------
    result : tuple
        A tuple of a transformed and normalized matrix and the global mean profile.
        matrix : np.ndarray
            The transformed and normalized matrix (n_subjects, n_voxels). Each row is a subject.
        gmp : np.ndarray
            The global mean profile.
    """
    if log_transform:
        matrix = log_voxels(matrix)
    logger.debug(f'Log transformed matrix shape: {matrix.shape}')
    matrix, gmp = normalize(matrix, seed_mask)
    logger.debug(f'Normalized matrix shape: {matrix.shape}')
    return matrix, gmp


def subtract_gmp(subjects_data: np.ndarray, gmp: np.ndarray,
                 seed_mask: np.ndarray) -> np.ndarray:
    """
    Subtract the global mean profile (GMP) from each subject's data.

    Parameters
    ----------
    subjects_data : np.ndarray
        2D array of shape (n_subjects, n_voxels). Each row is a subject.
    gmp : np.ndarray
        1D array of shape (n_voxels,). The global mean profile.

    Returns
    -------
    result : np.ndarray
        2D array of shape (n_subjects, n_voxels). The result of subtracting the GMP from each subject's data.
    """

    check_data.check_type(subjects_data, np.ndarray)
    check_data.check_type(gmp, np.ndarray)
    check_data.check_dimensions_number(gmp, 1)
    if (subjects_data.ndim == 1):
        subjects_data = np.expand_dims(subjects_data, axis=0)
    check_data.check_dimensions(gmp, (subjects_data.shape[1],))
    check_data.check_dimensions_number(subjects_data, 2)
    result = np.zeros(subjects_data.shape)
    subjects_data = np.apply_along_axis(
        subtract_mean, 1, subjects_data, seed_mask)
    for i in range(subjects_data.shape[0]):
        result[i, :] = subjects_data[i, :] - gmp
    return result
