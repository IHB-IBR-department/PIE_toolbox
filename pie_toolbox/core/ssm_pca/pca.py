import numpy as np
import logging
from pie_toolbox.core.ssm_pca import utils, log_norm
from pie_toolbox.core.common.logger import setup_root_logger, get_logger, section
from pie_toolbox.core.pattern_handler import scores
from pie_toolbox.core.common import converters
setup_root_logger('ssm_pca')
logger = get_logger('ssm_pca')


def ssm_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs SSM-PCA with normalized data.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (n_subjects, n_voxels). Each row is a subject.

    Returns
    -------
    patterns : np.ndarray
        The projected data using weighted eigenvectors.
        Shape (n_subjects, n_voxels).

    eigvalues_sorted : np.ndarray
        The sorted eigenvalues.
    """

    section('Preprocessing Initial Data', logger=logger, level=logging.DEBUG)
    logger.debug(
        'Preprocessing part in progress -> skipping using inital matrix')

    logger.debug('Axis 0 - Subjects\nAxis 1 - Voxels')
    logger.debug(f'Initial matrix shape: {matrix.shape}')

    section(
        'Obtaining Subject-by-Subject Covariance Matrix',
        logger=logger,
        level=logging.DEBUG)
    cov_matrix = utils.get_sub_by_sub_covmat(matrix)

    section(
        'Computing Eigen Decomposition of Covariance Matrix',
        logger=logger,
        level=logging.DEBUG)
    eigvalues_sorted, eigvectors_sorted = utils.compute_sorted_eigenpairs(
        cov_matrix)

    section(
        'Weighting Eigenvectors by Corresponding Eigenvalues',
        logger=logger,
        level=logging.DEBUG)
    weighted_vectors = utils.weight_eigenvectors_by_eigenvalues(
        eigvectors_sorted, eigvalues_sorted, logger=logger)
    weighted_vectors = eigvectors_sorted

    section(
        'Projecting Data Using Weighted Eigenvectors',
        logger=logger,
        level=logging.DEBUG)
    patterns = matrix.T @ weighted_vectors
    logger.debug(f'patternsing matrix shape: {patterns.shape}')
    logger.debug(f'patternsing matrix: {patterns}')

    # Transpose the patterns to make them (n_subjects, n_voxels)
    return patterns.T, eigvalues_sorted
