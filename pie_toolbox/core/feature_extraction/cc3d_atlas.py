import numpy as np
import cc3d
import pie_toolbox.core.validators.check_data as check_data
from pie_toolbox.core.common import thesholding


def atlas_delta(atlas: np.ndarray, delta: float = 0.05,
                connectivity: int = 6) -> np.ndarray:
    """
    Apply a delta value to the atlas, splitting touching regions into separate regions.

    Parameters
    ----------
    atlas : np.ndarray
        The 3D atlas to process.
    delta : float, optional
        The delta value to apply. Defaults to 0.05.
    connectivity : int, optional
        The number of neighbor voxels to consider when finding connected components. Defaults to 6.

    Returns
    -------
    np.ndarray
        The atlas with touching regions split.
    """
    atlas_cc = cc3d.connected_components(
        atlas, delta=delta, connectivity=connectivity)  # 6-connected
    return atlas_cc


def atlas_dust(atlas: np.ndarray, dust: int = 200,
               connectivity: int = 6) -> np.ndarray:
    """
    Remove small regions from atlas.

    Parameters
    ----------
    atlas : np.ndarray
        The 3D atlas to process.
    dust : int, optional
        The size threshold for regions to be removed. Defaults to 200.
    connectivity : int, optional
        The number of neighbor voxels to consider when finding connected components. Defaults to 6.

    Returns
    -------
    np.ndarray
        The atlas with small regions removed.
    """
    atlas_dusted = cc3d.dust(
        atlas, threshold=dust,
        connectivity=connectivity, in_place=True,
    )
    return atlas_dusted


def atlas_renumber(atlas: np.ndarray):
    """
    Renumber regions in atlas to integers.

    Parameters
    ----------
    atlas : np.ndarray
        The 3D atlas to renumber.

    Returns
    -------
    atlas_renumbered : np.ndarray
        The renumbered atlas.
    """
    atlas_renumbered = np.zeros_like(atlas, dtype=int)
    atlas_uniq = np.unique(atlas)
    atlas_uniq = atlas_uniq[atlas_uniq != 0]
    for i, region in enumerate(atlas_uniq, start=1):
        atlas_renumbered[atlas == region] = i
    return atlas_renumbered


def reshape_atlas(atlas: np.ndarray, shape: tuple = None):
    if len(atlas.shape) != 3:
        if shape is None:
            raise ValueError("Shape must be provided if atlas is not 3D")
        elif len(shape) != 3:
            raise ValueError("Shape must be 3D")
        else:
            atlas = atlas.reshape(shape)
    return atlas


def get_atlas(atlas: np.ndarray, shape: tuple = None, threshold_percentage: float = 0.05,
              delta: float = 0.05, dust: int = 200, connectivity: int = 6):
    """
    Process the given 3D or 1D atlas by reshaping, thresholding, applying delta, removing small regions, renumbering, and flattening.

    Parameters
    ----------
    atlas : np.ndarray
        The 3D or 1D atlas to process.
    shape : tuple, optional
        The desired shape to reshape the atlas if it is not already 3D.
    threshold_percentage : float, optional
        The percentage of the maximum value to threshold the atlas. Defaults to 0.05.
    delta : float, optional
        The delta value to apply for splitting touching regions. Defaults to 0.05.
    dust : int, optional
        The size threshold for regions to be removed. Defaults to 200.
    connectivity : int, optional
        The number of neighbor voxels to consider when finding connected components. Defaults to 6.

    Returns
    -------
    np.ndarray
        The processed and flattened atlas (1D).
    """

    atlas = reshape_atlas(atlas, shape=shape)
    atlas = thesholding.threshold_relative(
        atlas, threshold_percentage, absolute=True)
    atlas = atlas_delta(atlas, delta=delta, connectivity=connectivity)
    atlas = atlas_dust(atlas, dust=dust, connectivity=connectivity)
    atlas = atlas_renumber(atlas)
    atlas = np.ravel(atlas)
    return atlas
