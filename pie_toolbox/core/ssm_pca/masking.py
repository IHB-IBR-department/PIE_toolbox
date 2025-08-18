import numpy as np
from pie_toolbox.core.common import converters


def get_threshold_mask(
    matrix: np.ndarray,  # 0 - subjects, 1 - voxels
    threshold: float
) -> np.ndarray:

    mask = np.empty(shape=matrix.shape, dtype=bool)
    mask.fill(True)

    mask = np.abs(matrix) >= np.nanmax(
        np.abs(matrix), axis=1)[
        :, None] * threshold
    mask[np.isnan(matrix)] = False

    mask_vector = np.prod(mask, axis=0, dtype=bool)

    return mask_vector


def get_loaded_mask(
    mask_voxels: np.ndarray,
    threshold: float = 0.5,
    is_threshold_relative: bool = False
) -> np.ndarray:

    if is_threshold_relative:
        threshold = np.nanmax(mask_voxels) * threshold

    return (mask_voxels >= threshold)


def get_atlas_mask(
    atlas_voxels: np.ndarray,
    indexes: list = None,
) -> np.ndarray:
    indexes = converters.convert_index(indexes)
    return np.isin(atlas_voxels, indexes)


def mask_image(
        subject_matrix: np.ndarray,  # 0 - subjects, 1 - voxels

        threshold: float,

        mask_loaded=None,
        mask_loaded_threshold=0.5,
        mask_loaded_is_threshold_relative=False,

    atlas_loaded=None,
    indexes_mask=None
):
    '''
    Input:
        subject_matrix: 0 - subjects, 1 - voxels
        parameters for:
            threshold mask
            loaded mask
            atlas mask

    Output:
        voxels (masked)
        final mask
    '''

    mask_threshold = get_threshold_mask(subject_matrix, threshold=threshold)

    combined_mask = mask_threshold

    if (mask_loaded is not None):
        mask_loaded = get_loaded_mask(
            mask_loaded,
            threshold=mask_loaded_threshold,
            is_threshold_relative=mask_loaded_is_threshold_relative)
        if mask_loaded is not None:
            combined_mask &= mask_loaded
    if (atlas_loaded is not None):
        mask_atlas = get_atlas_mask(atlas_loaded, indexes=indexes_mask)
        if mask_atlas is not None:
            combined_mask &= mask_atlas

    voxels = subject_matrix[:, combined_mask]

    return voxels, combined_mask
