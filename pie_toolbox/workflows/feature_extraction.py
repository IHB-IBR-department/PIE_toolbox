from dataclasses import dataclass
import numpy as np

from pie_toolbox.core.common.logger import setup_root_logger, get_logger
from pie_toolbox.core.common.file_tools import load_subjects, open_image, save_images
from pie_toolbox.core.ssm_pca.masking import mask_image
from pie_toolbox.core.data_classes.image_params import ImageParams
from pie_toolbox.core.ssm_pca import log_norm
from pie_toolbox.core.validators import check_data

from pie_toolbox.core.common import file_tools

from pie_toolbox.core.feature_extraction import cc3d_atlas


class AtlasVOI:
    raw_pattern: np.ndarray = None
    atlas: np.ndarray = None
    _image_params: ImageParams = None

    def __init__(self):
        pass

    def get_pattern(self, voxelpca, index: int):
        from pie_toolbox.workflows.ssm_pca import VoxelPCA
        """
        Set the raw pattern from the given VoxelPCA object.

        Parameters
        ----------
        voxelpca : VoxelPCA
            The VoxelPCA object to get the pattern from.
        index : int
            The index of the pattern to get.

        Notes
        -----
        The raw pattern is reshaped from the 1D array to 3D using self._image_params.shape
        and self._image_params.zeros_mask.
        """
        if self._image_params is None:
            self._image_params = voxelpca._image_params
        raw_pattern_voxels = voxelpca.patterns[0]
        self.raw_pattern = file_tools.reshape_image(
            raw_pattern_voxels,
            self._image_params.shape,
            self._image_params.zeros_mask)

    def get_atlas(self, threshold_percentage: float = 0.05, delta: float = 0.05, dust: int = 200,
                  pattern: np.ndarray = None, shape: tuple = None, connectivity: int = 6):
        """
        Process the given pattern by reshaping, thresholding, applying delta, removing small regions, renumbering, and flattening.

        Parameters
        ----------
        pattern : np.ndarray
            The 3D pattern to process. If None, self.raw_pattern will be used.
        shape : tuple, optional
            The desired shape to reshape the pattern if it is not already 3D.
        threshold_percentage : float, optional
            The percentage of the maximum value to threshold the pattern. Defaults to 0.05.
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
        if pattern is None:
            check_data.check_not_none_or_empty(self.raw_pattern)
            pattern = self.raw_pattern
        atlas = cc3d_atlas.get_atlas(
            pattern,
            shape=shape,
            threshold_percentage=threshold_percentage,
            delta=delta,
            dust=dust,
            connectivity=connectivity)
        self.atlas = atlas[self._image_params.zeros_mask]
        return self.atlas
