import numpy as np
from pie_toolbox.core.common.logger import get_logger

logger = get_logger('ssm_pca.utils')


def get_sub_by_sub_covmat(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a subject-by-subject similarity matrix using dot-product similarity.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (n_subjects, n_voxels). Each row corresponds to a subject,
        and each column corresponds to a voxel.

    Returns
    -------
    cov_matrix : np.ndarray
        Matrix of shape (n_subjects, n_subjects), where each element [i, j] is the
        unnormalized dot-product similarity between subject i and subject j across all voxels.

    Notes
    -----
    This function computes a similarity matrix using the dot product of the input matrix
    with its transpose: matrix @ matrix.T. The result is not a true covariance matrix,
    as the data is not mean-centered. However, it reflects relative similarity between
    subjects based on shared voxel activation patterns.
    """

    cov_mat = matrix @ matrix.T  # shape: must be  (n_subjects, n_subjects)

    if logger is not None:
        # shape: must be (n_subjects, n_voxels)
        logger.debug(f'Initial matrix shape: {matrix.shape}')
        logger.debug(f'cov_mat matrix shape: {cov_mat.shape}')

    return cov_mat


def compute_sorted_eigenpairs(
        matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and return eigenvalues and eigenvectors of a symmetric matrix,
    sorted in descending order of eigenvalues.

    Parameters
    ----------
    matrix : np.ndarray
        A symmetric 2D square matrix (e.g., a covariance matrix) of shape (n, n).
        Assumes real-valued input.

    Returns
    -------
    eigvalues : np.ndarray
        1D array of eigenvalues sorted in descending order, shape (n,).

    eigvectors : np.ndarray
        2D array where each column is an eigenvector corresponding to the sorted
        eigenvalues, shape (n, n). Eigenvectors are normalized and orthogonal.

    Notes
    -----
    This function uses `np.linalg.eigh`, which is optimized for symmetric
    (or Hermitian) matrices and guarantees real-valued outputs.
    """
    assert np.allclose(matrix, matrix.T), "Input matrix must be symmetric"

    eigvalues, eigvectors = np.linalg.eig(matrix)

    # Sort eigenvalues and eigenvectors in descending order
    indexes = eigvalues.argsort()[::-1]
    eigvalues = eigvalues[indexes]
    eigvectors = eigvectors[:, indexes]

    if logger is not None:
        logger.debug(f'Eigenvalues shape: {eigvalues.shape}')
        logger.debug(f'Eigenvectors shape: {eigvectors.shape}')
        logger.debug(f'Eigenvalues sorted: {eigvalues}')
        logger.debug(f'Eigenvectors sorted: {eigvectors}')

    return eigvalues, eigvectors


def weight_eigenvectors_by_eigenvalues(
    eigvectors: np.ndarray,
    eigvalues: np.ndarray,
    logger=None
) -> np.ndarray:
    """
    Weight eigenvectors by the square root of their corresponding eigenvalues
    using element-wise broadcasting.

    Parameters
    ----------
    eigvectors : np.ndarray
        2D array of eigenvectors, shape (n_features, n_components).
        Each column corresponds to one eigenvector.

    eigvalues : np.ndarray
        1D array of eigenvalues, shape (n_components,).
        Assumed to be sorted in descending order.

    logger : logging.Logger, optional
        Logger instance for debug output. If None, logging is skipped.

    Returns
    -------
    weighted_vectors : np.ndarray
        Eigenvectors weighted by sqrt of eigenvalues, same shape as eigvectors.
        Each eigenvector column is multiplied elementwise by sqrt of its eigenvalue.

    Notes
    -----
    Weighting eigenvectors by the square root of eigenvalues is often used in PCA
    to scale the components by their explained variance (standard deviation).

    The function assumes that eigvalues are non-negative.
    """
    sqrt_eigvalues = np.sqrt(eigvalues)
    weighted_vectors = eigvectors * sqrt_eigvalues[np.newaxis, :]

    if logger is not None:
        logger.debug(f'Square roots of eigenvalues: {sqrt_eigvalues}')
        logger.debug(
            f'Eigenvectors shape before weighting: {eigvectors.shape}')
        logger.debug(f'Weighted eigenvectors shape: {weighted_vectors.shape}')

    return weighted_vectors


def subtract_mean(arr: np.ndarray, mask: np.ndarray = None):
    """
    Subtracts the mean of the array or masked array from each element.

    Parameters
    ----------
    arr : numpy.ndarray
        1D array from which the mean is subtracted.
    mask : numpy.ndarray, optional
        Boolean mask array for the seed region. If not None, normalize by the mean of the seed region.

    Returns
    -------
    result : numpy.ndarray
        The normalized array.
    """

    arr_val = arr[:]
    if not (mask is None):
        arr_val = arr_val[mask]
    mean_val = np.mean(arr_val)
    result = arr[:] - mean_val
    return result


def get_explained_variance(eigvalues: np.ndarray) -> np.ndarray:
    """
    Computes the explained variance for each eigenvalue.

    Parameters
    ----------
    eigvalues : np.ndarray
        1D array of eigenvalues, shape (n_components,).
        Assumed to be sorted in descending order.

    Returns
    -------
    explained_variance : np.ndarray
        1D array of explained variance for each eigenvalue, shape (n_components,).
    """
    return eigvalues / np.sum(eigvalues)


def invert_patterns(patterns: np.ndarray,
                    scores: np.ndarray, labels: np.ndarray):
    if (np.unique(labels).shape[0] == 2):
        labels_binary = labels == labels[0]
        for i_pattern in range(patterns.shape[0]):
            scores_0 = scores[labels_binary, i_pattern]
            scores_1 = scores[~labels_binary, i_pattern]

            if (np.mean(scores_0) < np.mean(scores_1)):
                patterns[i_pattern, :] *= -1
                scores[:, i_pattern] *= -1

    return patterns, scores
