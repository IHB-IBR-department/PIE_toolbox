
from dataclasses import dataclass
import numpy as np

from pie_toolbox.core.common.logger import setup_root_logger, get_logger
from pie_toolbox.core.common.file_tools import load_subjects, open_image, save_images
from pie_toolbox.core.common.converters import get_scores_from_list, get_whole_brain_scores
from pie_toolbox.core.ssm_pca.masking import mask_image
from pie_toolbox.core.data_classes.image_params import ImageParams
from pie_toolbox.core.ssm_pca import log_norm, pca
import pie_toolbox.core.ssm_pca.utils as pca_utils
from pie_toolbox.core.pattern_handler import combine_patterns, scores
from pie_toolbox.core.validators import check_data
from pie_toolbox.workflows.image_dataset import ImageDataset
from pie_toolbox.workflows.feature_extraction import AtlasVOI
import pie_toolbox.core.common.stats as stats
import pie_toolbox.core.common.converters as converters

setup_root_logger('ssm_pca_workflow')
logger = get_logger('ssm_pca_workflow')


class VoxelPCA:
    patterns: np.ndarray = None
    explained_variance: np.ndarray = None
    _image_params = ImageParams()
    _seed_regions = None
    _scores_original = None
    _labels_original = None

    def __init__(self, patterns: np.ndarray = None, explained_variance: np.ndarray = None,
                 _image_params: ImageParams = None, _seed_regions=None, _scores_original=None, _labels_original=None):
        self.patterns = patterns
        self.explained_variance = explained_variance
        self._image_params = _image_params
        self._seed_regions = _seed_regions
        self._scores_original = _scores_original
        self._labels_original = _labels_original

    def fit(self, images: ImageDataset | np.ndarray,
            labels=None, log_transform=True):
        """
        Fit the VoxelPCA model to the data.

        Parameters
        ----------
        images : ImageDataset or np.ndarray
            The data to fit the model to.
            If ImageDataset, the subject_images attribute will be used.
            If np.ndarray, the data is expected to be of shape (n_subjects x n_voxels).
        labels : list, optional
            A list of labels for the subjects.
        log_transform : bool, optional
            Whether to apply a logarithmic transformation to the data (if ImageDataset was not normalized). Default is True.

        Raises
        ------
        TypeError
            If images is not ImageDataset or np.ndarray.
        ValueError
            If images is ImageDataset and subject_images is None.

        Updates
        -------
        patterns : np.ndarray
            The resulting patterns from fitting the model.

        Notes
        -----
        If images is ImageDataset and not normalized, it will be normalized using ImageDataset.log_normalize().
        """

        if isinstance(images, ImageDataset):
            if images.normalized_subject_images is not None:
                image_matrix = images.normalized_subject_images
            else:
                logger.warning(
                    'Images are not normalized. Normalizing now with ImageDataset.log_normalize()')
                images.log_normalize(log_transform=log_transform)
                image_matrix = images.normalized_subject_images
        elif isinstance(images, np.ndarray):
            image_matrix = images
        else:
            raise TypeError(
                'Images must be either ImageDataset or np.ndarray (n_subjects x n_voxels)')

        logger.debug(f'Image matrix shape: {image_matrix.shape}')
        logger.debug(f'Image matrix: {image_matrix}')

        if images.labels is not None:
            labels = images.labels

        check_data.check_dimensions_number(image_matrix, 2)
        patterns, eigenvals = pca.ssm_pca(image_matrix)

        # Compute scores
        scores_list = scores.get_scores(image_matrix, patterns)
        scores_array = converters.get_scores_from_list(
            scores_list, region_index=0)
        # Invert patterns if 1st group has lower mean than 2nd
        self.patterns, scores_array = pca_utils.invert_patterns(
            patterns, scores_array, labels)

        self.explained_variance = pca_utils.get_explained_variance(eigenvals)

        self._image_params = images._image_params

        self._scores_original = scores_array
        logger.info(f'Original scores shape: {self._scores_original.shape}')
        logger.info(f'Original labels shape: {labels.shape}')
        self._labels_original = images.labels

    def save_patterns(self, filepath, name='Pattern'):
        """
        Saves the PCA patterns to NIfTI images.

        Parameters
        ----------
        filepath : str
            The directory path where the patterns will be saved.
        name : str, optional
            The base name for the saved files. Default is 'Pattern'.

        Notes
        -----
        The shape, affine transformation, and zeros mask are taken
        from the `_image_params` attribute.
        """

        save_images(self.patterns, filepath=filepath, filename=name,
                    shape=self._image_params.shape,
                    affine=self._image_params.affine,
                    zeros_mask=self._image_params.zeros_mask)

    def get_scores(self, images: ImageDataset | np.ndarray,
                   atlases: AtlasVOI | np.ndarray = None) -> list:
        """
        Compute the scores of the PCA patterns in the provided image data, optionally using specific atlases.

        Parameters
        ----------
        images : ImageDataset or np.ndarray
            The image data to compute scores from. If ImageDataset, the subject_images attribute will be used.
            If np.ndarray, the data is expected to be of shape (n_subjects, n_voxels).
        atlases : np.ndarray or AtlasVOI, optional
            The atlases in which to compute the scores
            If np.ndarray, the data is expected to be of shape (n_patterns, n_voxels).
            If AtlasVOI, the atlas attribute will be used.
            If None, the scores are computed over the whole brain (n_regions = 1).

        Returns
        -------
        list
            A list with n_patterns elements, each element is an array of arrays of shape (n_subjects, n_regions), each containing the scores of the given pattern in the given data.

        Raises
        ------
        ValueError
            If images is an ImageDataset and subject_images is None.
        TypeError
            If images is not an ImageDataset or np.ndarray.
        """

        if isinstance(images, ImageDataset):
            if images.normalized_subject_images is None:
                raise ValueError(
                    'Images must be normalized. Use log_normalize or log_normalize_for_new_images.')
            else:
                image_matrix = images.normalized_subject_images
        elif isinstance(images, np.ndarray):
            image_matrix = images
        else:
            raise TypeError(
                'Images must be either ImageDataset or np.ndarray (n_subjects x n_voxels)')

        if atlases is not None:
            if isinstance(atlases, AtlasVOI):
                atlases = atlases.atlas.reshape(1, -1)
            elif isinstance(atlases, np.ndarray):
                check_data.check_type(atlases, np.ndarray)
                if len(atlases.shape) == 1:
                    atlases = atlases.reshape(1, -1)
            else:
                raise TypeError(
                    'Atlases must be either AtlasVOI or np.ndarray (n_patterns x n_voxels)')

        scores_list = scores.get_scores(
            image_matrix, patterns=self.patterns, atlases=atlases)
        return scores_list

    def get_patterns(self, explained_variance=0,
                     cumulative_explained_variance=1, n_patterns=100):
        """
        Retrieve a subset of patterns based on explained variance criteria.

        Parameters
        ----------
        explained_variance : float, optional
            Minimum explained variance threshold for patterns. Default is 0.
        cumulative_explained_variance : float, optional
            Maximum cumulative explained variance threshold for patterns. Default is 1.
        n_patterns : int, optional
            Maximum number of patterns to initially select for further filtering. Default is 100.

        Returns
        -------
        VoxelPCA
            A new VoxelPCA object containing patterns that meet the specified explained variance criteria.
        """

        bool_array = np.ones(self.patterns.shape[0], dtype=bool)
        # Get first n_patterns patterns
        if n_patterns < self.patterns.shape[0]:
            bool_array[n_patterns:] = False
        # Get patterns that > explained_variance
        bool_array = np.logical_and(
            bool_array, self.explained_variance > explained_variance)
        # Get patterns that < cumulative_explained_variance
        cumulative_explained_variance_array = np.cumsum(
            self.explained_variance)
        if cumulative_explained_variance_array[0] > cumulative_explained_variance:
            bool_array_temp = np.ones(self.patterns.shape[0], dtype=bool)
            bool_array_temp[1:] = False
            bool_array = np.logical_and(bool_array, bool_array_temp)
        else:
            bool_array = np.logical_and(
                bool_array,
                cumulative_explained_variance_array <= cumulative_explained_variance)
        if bool_array.sum() == 0:
            logger.warning(
                'No patterns meet the specified explained variance criteria.')
        return VoxelPCA(patterns=self.patterns[bool_array, :],
                        explained_variance=self.explained_variance[bool_array],
                        _image_params=self._image_params,
                        _seed_regions=self._seed_regions,
                        _scores_original=self._scores_original[:, bool_array],
                        _labels_original=self._labels_original)

    def get_patterns_by_indexes(self, indexes: int | list = 0):
        """
        Return a new VoxelPCA object with the specified patterns from the current object.

        Parameters
        ----------
        indexes : int or list of int, optional
            The index or indexes of the patterns to select. If int, a single pattern is selected.
            If list of int, all patterns with the given indexes are selected. Default is 0.

        Returns
        -------
        VoxelPCA
            A new VoxelPCA object with the selected patterns.
        """
        if isinstance(indexes, int):
            indexes = [indexes]
        return VoxelPCA(patterns=self.patterns[indexes, :],
                        explained_variance=self.explained_variance[indexes],
                        _image_params=self._image_params,
                        _seed_regions=self._seed_regions,
                        _scores_original=self._scores_original[:, indexes],
                        _labels_original=self._labels_original)

    def get_combined_pattern(
            self, scores: list | np.ndarray, labels: np.ndarray):
        """
        Combine the patterns to a single pattern using logistic regression coefficients.

        Parameters
        ----------
        scores : list|np.ndarray
            The scores of the patterns to combine. If list, it is expected to be a list of arrays with shape (n_subjects, 1).
            If np.ndarray, it is expected to be an array with shape (n_subjects) or (n_subjects, 1).
        labels : np.ndarray
            The labels of the subjects. Shape (n_subjects,)

        Returns
        -------
        combined_patterns : VoxelPCA
            The combined pattern.
        """

        # Check if scores are 1D and convert to 1D numpy array if necessary
        scores = get_whole_brain_scores(scores)
        check_data.check_type(labels, np.ndarray)

        coefficients, _ = combine_patterns.logreg_pattern_coefficients(
            scores, labels)
        combined_pattern = combine_patterns.combine_patterns(
            self.patterns, coefficients)
        combined_pattern = np.reshape(combined_pattern, (1, -1))

        new_scores = np.dot(scores, coefficients)
        new_scores = np.reshape(new_scores, (1, -1))

        return VoxelPCA(patterns=combined_pattern, explained_variance=np.array([1]),
                        _image_params=self._image_params,
                        _seed_regions=self._seed_regions,
                        _scores_original=new_scores,
                        _labels_original=labels)
