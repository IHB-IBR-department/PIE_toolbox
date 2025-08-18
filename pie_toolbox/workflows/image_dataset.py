from dataclasses import dataclass
import numpy as np

from pie_toolbox.core.common.logger import setup_root_logger, get_logger
from pie_toolbox.core.common.file_tools import load_subjects, open_image, save_images
from pie_toolbox.core.ssm_pca.masking import mask_image
from pie_toolbox.core.data_classes.image_params import ImageParams
from pie_toolbox.core.ssm_pca import log_norm
from pie_toolbox.core.validators import check_data

setup_root_logger('ssm_pca_workflow')
logger = get_logger('ssm_pca_workflow')


@dataclass
class ImageDataset():

    '''
    Class for managing image files
    '''

    _image_params = ImageParams()

    # IMAGE FILES
    raw_subject_images: np.ndarray = None
    masked_subject_images: np.ndarray = None
    normalized_subject_images: np.ndarray = None
    labels: np.ndarray = None
    loaded_mask: np.ndarray = None
    atlas_mask: np.ndarray = None

    def add_group(self, files, label='train'):
        """
        Adds a new group of image files to the dataset.

        Parameters
        ----------
        files : list
            A list of file paths to the image files to be added.
        label : str, optional
            A label for the group of images. Default is 'train'.

        Updates
        -------
        raw_subject_images : np.ndarray
            The concatenated image data matrix.
        labels : np.ndarray
            The extended array of labels for the image data.
        """

        matrix, self._image_params.shape, self._image_params.affine = load_subjects(
            files, shape=self._image_params.shape, affine=self._image_params.affine)
        if self.labels is None:
            self.labels = np.array([], dtype=str)
        new_labels = np.array([label] * matrix.shape[0], dtype=str)
        self.labels = np.concatenate((self.labels, new_labels))
        if self.raw_subject_images is None:
            self.raw_subject_images = matrix
        else:
            # logger.debug(self.raw_subject_images.shape)
            # logger.debug(matrix.shape)
            self.raw_subject_images = np.concatenate(
                (self.raw_subject_images, matrix), axis=0)
        logger.info(f'Loaded {matrix.shape[0]} images for {label} group')

    def log_normalize(self, log_transform: bool = True):
        """
        Apply a logarithmic transformation to the image data, and normalize it.

        Parameters
        ----------
        log_transform : bool, optional
            Whether to apply a logarithmic transformation to the data. Default is True.

        Updates
        -------
        subject_images : np.ndarray
            The transformed and normalized image data.

        Raises
        ------
        ValueError
            If no images are available to normalize.

        Notes
        -----
        If masked images are available, they are used for normalization.
        Otherwise, a warning is issued and raw images are used.
        """

        if (self.masked_subject_images is not None):
            images_to_normalize = self.masked_subject_images
        elif (self.raw_subject_images is not None):
            logger.warning(
                'No mask was applied. Using raw images for normalization.')
            images_to_normalize = self.raw_subject_images
        else:
            raise ValueError(
                'No images to normalize. Use add_group to add images and apply_mask if needed.')
        self.normalized_subject_images, self._image_params.gmp = log_norm.log_normalize(
            images_to_normalize, log_transform=log_transform)

    def normalize_for_new_images(self, matrix=None, gmp=None):
        """
        Normalizes the image data using the global mean profile (GMP) obtained during the previous normalization.

        Use this function if you want to apply the same normalization to new images.

        Parameters
        ----------
        gmp : array_like, optional
            The global mean profile to use for normalization. If not provided, the GMP from the previous normalization is used.

        Raises
        ------
        ValueError
            If no images have been loaded or if no GMP was provided or set previously.

        Updates
        -------
        normalized_subject_images : np.ndarray
            The normalized image data matrix.
        """
        if matrix is None:
            if self.masked_subject_images is not None:
                matrix = self.masked_subject_images
            elif self.raw_subject_images is not None:
                logger.warning(
                    'No mask was applied. Using raw images for normalization.')
                matrix = self.raw_subject_images
            else:
                raise ValueError(
                    'No images to normalize. Use add_group to add images and apply_mask if needed.')

        if gmp is not None:
            self._image_params.gmp = gmp
        elif self._image_params.gmp is not None:
            gmp = self._image_params.gmp
        else:
            raise ValueError('No global mean profile (GMP) was provided.')

        self.normalized_subject_images = log_norm.subtract_gmp(
            matrix, gmp, seed_mask=self._image_params.seed_mask)

        return self.normalized_subject_images

    def load_mask(self, filepath):
        """
        Loads a mask image from the specified file path.

        Parameters
        ----------
        filepath : str
            The file path to the mask image file.

        Updates
        -------
        loaded_mask : np.ndarray
            The loaded mask image as a numpy array.

        """

        self.loaded_mask, self._image_params.shape, self._image_params.affine = open_image(filepath=filepath,
                                                                                           shape=self._image_params.shape,
                                                                                           affine=self._image_params.affine,
                                                                                           ravel=True)
        logger.debug(f'Loaded mask of shape {self.loaded_mask.shape}')

    def load_atlas(self, filepath):
        """
        Loads an atlas image from the specified file path.

        Parameters
        ----------
        filepath : str
            The file path to the atlas image file.

        Updates
        -------
        atlas_mask : np.ndarray
            The loaded atlas image as a numpy array.

        """

        self.atlas_mask, self._image_params.shape, self._image_params.affine = open_image(filepath=filepath,
                                                                                          shape=self._image_params.shape,
                                                                                          affine=self._image_params.affine,
                                                                                          ravel=True)
        logger.debug(f'Loaded atlas of shape {self.atlas_mask.shape}')

    def apply_mask(self, threshold=0.5, loaded_mask_threshold=0.5,
                   loaded_mask_is_threshold_relative=False, indexes_mask=None):
        """
        Masks the image data using the specified parameters.

        Parameters
        ----------
        threshold : float, optional
            The threshold value for the mask. Default is 0.5.
        loaded_mask_threshold : float, optional
            The threshold value for the loaded mask. Default is 0.5.
        loaded_mask_is_threshold_relative : bool, optional
            Whether the loaded mask threshold is relative to the maximum value of the mask. Default is False.
        indexes_mask : list, optional
            The list of indexes to use for the atlas mask.

        Updates
        -------
        subject_images : np.ndarray
            The masked image data matrix.
        zeros_mask : np.ndarray
            The binary mask indicating which voxels were excluded.

        Notes
        -----
        This function uses self.raw_subject_images, self.loaded_mask (optional), and self.atlas_mask (optional).
        """
        self.masked_subject_images, self._image_params.zeros_mask = mask_image(self.raw_subject_images,
                                                                               threshold=threshold,
                                                                               mask_loaded=self.loaded_mask, mask_loaded_threshold=loaded_mask_threshold,
                                                                               mask_loaded_is_threshold_relative=loaded_mask_is_threshold_relative,
                                                                               atlas_loaded=self.atlas_mask,
                                                                               indexes_mask=indexes_mask)
        logger.debug(f'Masked images of shape {self.raw_subject_images.shape}')
        logger.debug(f'Excluded {self._image_params.zeros_mask.sum()} voxels')

    def adjust_to_dataset(self, original_dataset: 'ImageDataset'):
        """
        Adjusts the current dataset to the same parameters as another dataset.

        This function copies the image parameters from the original dataset and applies them to the current dataset.
        The current dataset is then log-normalized using the global mean profile from the original dataset.

        Parameters
        ----------
        original_dataset : ImageDataset
            The dataset to adjust to.

        Updates
        -------
        normalized_subject_images : np.ndarray
            The log-normalized image data matrix.
        _image_params : ImageParams
            The image parameters copied from the original dataset.
        """
        self._image_params = original_dataset._image_params
        matrix_masked = self.raw_subject_images[:,
                                                self._image_params.zeros_mask]
        matrix_log = log_norm.log_voxels(matrix_masked)
        self.normalized_subject_images = self.normalize_for_new_images(
            matrix_log, self._image_params.gmp)

    def get_info(self):
        '''
        Prints information about the loaded data to the logger.
        '''
        def is_loaded(variable):
            if variable is None:
                return 'None'
            return 'In class'

        logger.info('Loaded subjects: ')
        logger.debug(self.labels)
        labels_unique, labels_counts = np.unique(
            self.labels, return_counts=True)
        for label, count in zip(labels_unique, labels_counts):
            logger.info(f'{label}: {count}')
        logger.info(f'Image shape: {self._image_params.shape}')
        logger.info(f'Image affine: {is_loaded(self._image_params.affine)}')
        logger.info(f'Mask: {is_loaded(self.loaded_mask)}')
        logger.info(f'Atlas: {is_loaded(self.atlas_mask)}')
        if self.atlas_seed is None:
            logger.info('Seed not chosen')
        else:
            logger.info('Seed chosen')
