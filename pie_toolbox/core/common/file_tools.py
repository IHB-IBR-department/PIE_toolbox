import numpy as np
import nibabel as nib
import nilearn
import nilearn.image
from pie_toolbox.core.common.converters import files_list
import os
from pie_toolbox.core.common.logger import setup_root_logger, get_logger
from pie_toolbox.core.validators import check_data
from pie_toolbox.core.data_classes.image_params import ImageParams

setup_root_logger('file_tools')
logger = get_logger('file_tools')


def open_image(filepath: str, shape: tuple = None,
               affine: np.ndarray = None, ravel: bool = True) -> tuple:
    """
    Loads image from a file and optionally resamples it to a given shape and/or affine matrix.

    Parameters
    ----------
    filepath : str
        Path to the image file.
    shape : tuple, optional
        Shape of the output image.
    affine : np.ndarray, optional
        Affine matrix of the output image.
    ravel : bool, optional
        Whether to ravel the output image.

    Returns
    -------
    voxels : np.ndarray
        Raveled image voxels.
    shape : tuple
        Shape of the output image.
    affine : np.ndarray
        Affine matrix of the output image.
    """
    try:
        image = nib.load(filepath)
        # Resample
        if not (affine is None or shape is None):
            image = nilearn.image.resample_img(
                image,
                target_affine=affine,
                target_shape=shape,
                force_resample=True,
                copy_header=True)
        # Get voxels
        voxels = image.get_fdata()
        affine = image.affine
        shape = voxels.shape
        if ravel:
            voxels = np.ravel(voxels)
        voxels = np.nan_to_num(voxels)
        return voxels, shape, affine
    except BaseException:
        logger.error('Could not load image: {}'.format(filepath))
        return None, None, None


def reshape_image(voxels: np.ndarray, target_shape: tuple,
                  zeros_mask: np.ndarray) -> np.ndarray:
    """
    Reshape an image into the specified target shape using a zeros mask.

    Parameters
    ----------
    image : np.ndarray
        The 1D array representing the image that needs to be reshaped.
    target_shape : tuple
        The desired shape to which the image should be reshaped.
    zeros_mask : np.ndarray
        A boolean mask indicating the valid voxel positions in the target shape.

    Returns
    -------
    np.ndarray
        The image reshaped to the specified target shape.

    Raises
    ------
    ValueError
        If the product of the target_shape dimensions does not match the size of the zeros_mask.
    """
    check_data.check_type(voxels, np.ndarray)
    check_data.check_type(target_shape, tuple)
    check_data.check_type(zeros_mask, np.ndarray)
    if np.sum(zeros_mask) != voxels.size:
        logger.error('Zeros mask does not match image size')
    if np.prod(target_shape) != zeros_mask.size:
        logger.error(
            f'Cannot reshape array of size {zeros_mask.size} to {target_shape}. Please provide the correct mask for zero values.')

    intermed = np.zeros_like(zeros_mask, dtype=voxels.dtype)
    intermed[zeros_mask] = voxels
    return intermed.reshape(target_shape)


def load_subjects(files: list, shape: tuple = None,
                  affine: np.ndarray = None) -> tuple:
    """
    Load and process a list of image files into a 2D numpy array.

    Parameters
    ----------
    files : list
        A list of file paths to image files.
    shape : tuple, optional
        The desired shape for the images. If not provided, the shape will be determined from the first image.
    affine : np.ndarray, optional
        The affine transformation matrix for the images. If not provided, the affine will be determined from the first image.

    Returns
    -------
    subject_array : np.ndarray
        A 2D array (n_subjects, n_voxels) containing the processed images.
    image_shape : tuple
        The shape of the images.
    image_affine : np.ndarray
        The affine transformation matrix.
    """

    files = files_list(files)

    subject_array = []
    for i_file, file in enumerate(files):
        img_nifti, shape, affine = open_image(file, shape, affine, ravel=True)
        subject_array.append(img_nifti)

    subject_array = np.array(subject_array)
    if subject_array.ndim == 1:
        subject_array = subject_array[np.newaxis, :]

    logger.debug(f'Shape of subject array: {subject_array.shape}')

    return subject_array, shape, affine


def save_images(
    voxels: np.ndarray,
    filepath: str = "",
    filename: str = "image",
    shape: tuple = None,
    affine: np.ndarray = None,
    zeros_mask: np.ndarray = None,
    image_parameters: ImageParams = None
):
    """
    Save a 1D or 2D numpy array as NIfTI images.

    Parameters
    ----------
    voxels : np.ndarray
        The array to be saved. If 2D, each row will be saved as a separate image.
    filepath : str, optional
        The directory where the images will be saved. Default is current directory.
    filename : str, optional
        The base name of the saved images. Default is "image".
    shape : tuple, optional
        The desired shape for the saved images. Default is shape of the first image.
    affine : np.ndarray, optional
        The affine transformation matrix for the saved images. Default is affine of the first image.
    zeros_mask : np.ndarray, optional
        The mask of zero values for reshaping the images. Default is None.
    image_parameters : ImageParams, optional
        The image parameters to use for saving the images instead of shape and affine. Default is None.

    Returns
    -------
    None
    """
    if isinstance(voxels, list):
        voxels = np.array(voxels)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if filename.endswith('.nii'):
        filename = filename[:-4]
    if image_parameters is not None:
        shape = image_parameters.shape
        affine = image_parameters.affine
        zeros_mask = image_parameters.zeros_mask
    if voxels.ndim == 2:
        for i, row in enumerate(voxels):
            full_filepath = os.path.join(filepath, f"{filename}_{i}.nii")
            logger.debug(f'Saving image {i} to {full_filepath}')
            save_one_image(full_filepath, row, shape, affine, zeros_mask)
    else:
        full_filepath = os.path.join(filepath, f"{filename}.nii")
        save_one_image(full_filepath, voxels, shape, affine, zeros_mask)


def save_one_image(filepath: str, voxels: np.ndarray, shape: tuple = None,
                   affine: np.ndarray = None, zeros_mask: np.ndarray = None):
    """
    Save a 1D or 3D numpy array as a NIfTI image.

    Parameters
    ----------
    filepath : str
        The path to the saved image.
    voxels : np.ndarray
        The array to be saved. If 1D, it will be reshaped according to the shape and zeros_mask.
    shape : tuple, optional
        The desired shape for the saved image. Default is shape of the input array.
    affine : np.ndarray, optional
        The affine transformation matrix for the saved image. Default is identity matrix.
    zeros_mask : np.ndarray, optional
        The mask of zero values for reshaping the image. Default is None.

    Returns
    -------
    None
    """
    if zeros_mask is None:
        zeros_mask = np.ones(voxels.shape, dtype=bool)
    if voxels.ndim == 1:
        if shape is not None:
            voxels = reshape_image(voxels, shape, zeros_mask)
    check_data.check_dimensions_number(voxels, 3)
    check_data.check_type(voxels, np.ndarray)

    if affine is None:
        affine = np.eye(4)
        logger.warning('Affine matrix was not provided')
    image = nib.Nifti1Image(voxels, affine, dtype=voxels.dtype)
    nib.save(image, filepath)
    logger.info(f'Saved image: {filepath}')
