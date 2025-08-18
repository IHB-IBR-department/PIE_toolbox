import pie_toolbox.core.validators.check_data as check_data
import numpy as np
import scipy


def get_scores(data: np.ndarray, patterns: np.ndarray,
               atlases: np.ndarray = None) -> list:
    """
    Compute the scores of the given patterns in the given data, potentially limited to specific regions.

    Parameters
    ----------
    data : np.ndarray
        The data from which to compute the scores, shape (n_subjects, n_voxels).
    patterns : np.ndarray
        The patterns for which to compute the scores, shape (n_patterns, n_voxels).
    atlases : np.ndarray, optional
        The atlases in which to compute the scores, shape (n_patterns, n_voxels).
        If None, the scores are computed over the whole brain (n_regions = 1).

    Returns
    -------
    scores_list : list
        A list of arrays, each containing the scores of the given pattern in the given data, with shape (n_subjects, n_regions).
        The length of the list is the n_patterns.
    """
    check_data.check_type(data, np.ndarray)
    check_data.check_type(patterns, np.ndarray)
    if len(patterns.shape) == 1:
        patterns = patterns.reshape(1, -1)
    if atlases is not None:
        check_data.check_type(atlases, np.ndarray)
        if len(atlases.shape) == 1:
            atlases = atlases.reshape(1, -1)
    else:
        # Whole brain
        atlases = np.ones((patterns.shape[0], patterns.shape[1]))
    check_data.check_equal_axes((data, patterns), (1))
    check_data.check_equal_axes((patterns, atlases), (0))

    scores_list = []

    for i_pattern in range(patterns.shape[0]):
        # Get the scores for each subject: ndarray with shape (n_subjects,
        # n_regions) abd append to the list
        scores = []
        for i_subject in range(data.shape[0]):
            score = get_single_subject_scores(
                data[i_subject, :], patterns[i_pattern, :], atlases[i_pattern, :])
            scores.append(score)
        # scores = np.apply_along_axis(get_single_subject_scores, 1, data, patterns[i_pattern, :], atlases[i_pattern, :])
        scores_list.append(np.array(scores))

    return scores_list


def get_single_subject_scores(
        voxels: np.ndarray, pattern: np.ndarray, atlas: np.ndarray = None) -> np.ndarray:
    """
    Compute the scores of the given pattern in the given voxels, potentially limited to specific regions.

    Parameters
    ----------
    voxels : np.ndarray
        The voxels from which to compute the scores, shape (n_voxels,).
    pattern : np.ndarray
        The pattern for which to compute the scores, shape (n_voxels,).
    atlas : np.ndarray, optional
        The atlas in which to compute the scores, shape (n_voxels,). If None, the scores are computed over the whole brain.

    Returns
    -------
    scores : np.ndarray
        The scores of the given pattern in the given voxels, with shape (n_regions,)
    """
    check_data.check_type(voxels, np.ndarray)
    check_data.check_type(pattern, np.ndarray)
    check_data.check_type(atlas, np.ndarray)
    check_data.check_dimensions_number(voxels, 1)
    check_data.check_dimensions_number(pattern, 1)
    check_data.check_dimensions_number(atlas, 1)
    check_data.check_equal_axes((voxels, pattern), (0))

    # If atlas is None, get the whole brain
    if atlas is None:
        atlas = np.ones(pattern.shape[0])

    check_data.check_equal_axes((pattern, atlas), (0))
    # Get the unique regions
    region_ids = np.unique(atlas)
    region_ids = region_ids[region_ids != 0]
    scores = np.zeros((region_ids.shape[0]))

    # Compute the scores for each region
    for ind, region_id in enumerate(region_ids):
        pattern_masked = pattern.copy()
        pattern_masked[atlas != region_id] = 0
        scores[ind] = np.sum(np.multiply(voxels, pattern_masked))

    return scores
