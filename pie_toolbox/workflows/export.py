import numpy as np
from pie_toolbox.core.common import converters
from pie_toolbox.core.tables import save_table
from pie_toolbox.core.common import file_tools
from pie_toolbox.core.data_classes.image_params import ImageParams
from pie_toolbox.core.visualize import show_in_browser, plots
from pie_toolbox.workflows.ssm_pca import VoxelPCA
from pie_toolbox.core.classifier.metrics import Metrics
from pie_toolbox.core.classifier.cross_val import CrossValResults
from matplotlib import pyplot as plt

from pie_toolbox.core.common.logger import get_logger

logger = get_logger('export')


def _many_patterns_warning(
        patterns: np.ndarray, n_patterns: int = 10, supress: bool = False, axis: int = 0):
    if (supress == False) and patterns.shape[axis] > n_patterns:
        logger.warning(
            f"{patterns.shape[axis]} patterns found. Only the first {n_patterns} patterns will be exported.")
        return patterns[:n_patterns, :]
    else:
        return patterns


def show_patterns_in_browser(voxelpca: VoxelPCA, title: str = "Pattern",
                             threshold: float = 0.0, supress_warning: bool = False):
    """
    Show all patterns from the given VoxelPCA object in a new browser window.

    Parameters
    ----------
    voxelpca : VoxelPCA
        The VoxelPCA object containing the patterns.
    title : str, optional
        The title of the first pattern. Default is "Pattern".
    threshold : float, optional
        The relative threshold value for visualization. Defaults to 0.0.
    """

    all_patterns = _many_patterns_warning(
        voxelpca.patterns, supress=supress_warning)
    for i_pattern in range(all_patterns.shape[0]):
        pattern = all_patterns[i_pattern, :]
        image_3d = file_tools.reshape_image(
            pattern,
            voxelpca._image_params.shape,
            voxelpca._image_params.zeros_mask)
        show_in_browser.show_brain(
            image_3d,
            affine=voxelpca._image_params.affine,
            title=f"{title} {i_pattern+1}",
            threshold=threshold)


def plot_from_scores(scores: np.ndarray, labels: np.ndarray,
                     plot_type: str = "bar", figsize: tuple = (10, 4), sorted: bool = False):
    """
    Creates a plot of the given scores.

    Parameters
    ----------
    scores : np.ndarray
        Array of subject scores with shape (n_subjects,).
    labels : np.ndarray
        Array of labels for each subject with shape (n_subjects,).
    plot_type : str, optional
        The type of the plot to create. See `plots.get_plot` for available options. Default is "bar".
    figsize : tuple, optional
        Size of the figure in inches. Default is (10, 4).
    sorted : bool, optional
        If True, the groups are sorted by their median values. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot figure.
    """
    plot_fig = plots.get_plot(
        plot_type,
        scores,
        labels,
        figsize=figsize,
        sorted=sorted)
    return plot_fig


def plot_from_voxelpca(voxelpca: VoxelPCA, plot_type: str = "bar",
                       figsize: tuple = (10, 4), sorted: bool = False):
    """
    Creates a plot of the given VoxelPCA object's scores.

    Parameters
    ----------
    voxelpca : VoxelPCA
        The VoxelPCA object containing the scores.
    plot_type : str, optional
        The type of the plot to create. See `plots.get_plot` for available options. Default is "bar".
    figsize : tuple, optional
        Size of the figure in inches. Default is (10, 4).
    sorted : bool, optional
        If True, the groups are sorted by their median values. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot figure.
    """
    scores = voxelpca._scores_original
    labels = voxelpca._labels_original
    fig_list = []
    for i_pattern in range(scores.shape[1]):
        plot_fig = plot_from_scores(scores[:,
                                           i_pattern,
                                           ],
                                    labels,
                                    plot_type=plot_type,
                                    figsize=figsize,
                                    sorted=sorted)
        fig_list.append(plot_fig)
    return fig_list


def save_plot(figs: list | plt.Figure, output_path: str = "Plot"):
    """
    Saves the given figure or list of figures to the given output path.

    Parameters
    ----------
    figs : list or matplotlib.figure.Figure
        The figure or list of figures to save.
    output_path : str, optional
        The base path to save the figure(s) to. If no extension is given, '.png' is added.
        Default is "Plot.png".
    """
    if not isinstance(figs, list):
        figs = [figs]
    if output_path.endswith('.png'):
        output_path = output_path[:-4]
    for i, fig in enumerate(figs):
        path = output_path if len(figs) == 1 else f"{output_path}_{i+1}.png"
        path = converters.add_extension(path, 'png')
        fig.savefig(path)


def save_patterns(voxelpca: VoxelPCA, output_path: str = "",
                  filename: str = "Pattern", supress_warning: bool = False):
    """
    Saves the patterns from a VoxelPCA object to NIfTI images.

    Parameters
    ----------
    voxelpca : VoxelPCA
        The VoxelPCA object containing the patterns to be saved.
    output_path : str, optional
        The directory path where the images will be saved. Default is the current directory.
    filename : str, optional
        The base name for the saved image files. Default is 'Pattern'.

    Notes
    -----
    The image shape, affine transformation, and zeros mask are derived from the `voxelpca` object's `_image_params` attribute.
    """
    all_patterns = _many_patterns_warning(
        voxelpca.patterns, supress=supress_warning)

    file_tools.save_images(all_patterns, filename=filename, filepath=output_path,
                           shape=voxelpca._image_params.shape,
                           affine=voxelpca._image_params.affine,
                           zeros_mask=voxelpca._image_params.zeros_mask)


def export_metrics_xlsx(metrics: Metrics | CrossValResults,
                        output_path: str = "Table.xlsx", information_text: str = ""):
    """
    Saves the given metrics to an Excel file (.xlsx) using the given file path.

    Parameters
    ----------
    metrics : Metrics or CrossValResults
        The metrics object to be saved to the file from cross-validation.
    output_path : str, optional
        The path to save the file to. If no extension is given, '.xlsx' is added.
        Default is "Table.xlsx".
    information_text : str, optional
        Additional information to include in the file.
    """
    table = save_table.get_dataframe(metrics, information=information_text)
    save_table.save_excel(table, output_path)


def export_metrics_csv(metrics: Metrics | CrossValResults,
                       output_path: str = "Table.csv", information_text: str = ""):
    """
    Saves the given metrics to a CSV file using the given file path.

    Parameters
    ----------
    metrics : Metrics or CrossValResults
        The metrics object to be saved to the file from cross-validation.
    output_path : str, optional
        The path to save the file to. If no extension is given, '.csv' is added.
        Default is "Table.csv".
    information_text : str, optional
        Additional information to include in the file.
    """
    table = save_table.get_dataframe(metrics, information=information_text)
    save_table.save_csv(table, output_path)


def save_images(image: np.ndarray, filepath: str = "Image.nii",
                filename: str = "Image", image_parameters: ImageParams = None):
    """
    Saves a given images to a file using the given file path.

    Parameters
    ----------
    image : np.ndarray
        The image to be saved. If 2D, each row will be saved as a separate image.
    filepath : str, optional
        The path to save the file to. Default is "Image.nii".
    image_parameters : ImageParams, optional
        An ImageParams object containing information about the image's shape, affine transformation, and zeros mask.
        If not provided, the image is saved with the identity matrix as the affine transformation and no zeros mask.

    Notes
    -----
    The image is saved in NIfTI-1 format.
    """
    file_tools.save_images(
        image,
        filepath=filepath,
        filename=filename,
        image_parameters=image_parameters)


def save_image(image: np.ndarray, filepath: str = "Image.nii",
               image_parameters: ImageParams = None):
    """
    Saves a given image to a file using the given file path.

    Parameters
    ----------
    image : np.ndarray
        The image to be saved.
    filepath : str, optional
        The path to save the file to. Default is "Image.nii".
    image_parameters : ImageParams, optional
        An ImageParams object containing information about the image's shape, affine transformation, and zeros mask.
        If not provided, the image is saved with the identity matrix as the affine transformation and no zeros mask.

    Notes
    -----
    The image is saved in NIfTI-1 format.
    """
    file_tools.save_one_image(
        filepath, image, image_parameters=image_parameters)
