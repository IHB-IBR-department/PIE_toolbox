import pie_toolbox.core.visualize.colormap as colormap
from pie_toolbox.core.validators import check_data
import nibabel as nib
import nilearn.plotting
import numpy as np

from pie_toolbox.core.common.logger import setup_root_logger, get_logger

setup_root_logger('visualize')
logger = get_logger('visualize')


def show_brain(
    voxels: np.ndarray,
    affine: np.ndarray = None,
    title: str = "",
    threshold: float = 0,
    theshold_symmetric: bool = True,
    cmap: str = 'seismic',
    symmetric: bool = True
) -> None:
    """
    Visualize a brain image using nilearn's interactive viewer.

    Parameters
    ----------
    voxels : np.ndarray
        The voxel data for the brain image (3D).
    affine : np.ndarray, optional
        The affine transformation matrix for the image (4x4). Defaults to an identity matrix.
    title : str, optional
        The title for the displayed image. Defaults to an empty string.
    threshold : float, optional
        The relative threshold value for visualization. Defaults to 0.
    theshold_symmetric : bool, optional
        Whether the threshold is applied symmetrically. Defaults to True.
    cmap : str, optional
        The colormap for visualization. Defaults to 'seismic'.
    symmetric : bool, optional
        Whether the colormap is symmetric. Defaults to True.

    Returns
    -------
    None
    """

    check_data.check_range(threshold, 0, 1)
    check_data.check_type(voxels, np.ndarray)
    check_data.check_dimensions_number(voxels, 3)

    if affine is None:
        affine = np.eye(4)
    imag = nib.spatialimages.SpatialImage(voxels, affine)
    if cmap is None:
        cmap = colormap.discrete_colormap(len(np.unique(voxels)))
    if threshold == 0:
        threshold = (1e-20)
    else:
        if (theshold_symmetric):
            max_value = np.max(np.abs(voxels))
        else:
            max_value = np.max(voxels)
        threshold = max_value * threshold
    try:
        nilearn.plotting.view_img(imag,
                                  dim=2,
                                  title=title,
                                  threshold=threshold,
                                  resampling_interpolation='nearest',
                                  bg_img='MNI152',
                                  black_bg=False,
                                  cmap=cmap,
                                  symmetric_cmap=symmetric).open_in_browser()
    except BaseException:
        logger.warning('Failed to show image in browser')
