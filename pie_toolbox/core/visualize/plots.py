from matplotlib.figure import Figure
import pie_toolbox.core.visualize.colormap as colormap
from pie_toolbox.core.common import converters
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pie_toolbox.core.validators import check_data


def _check_scores(scores, labels):
    check_data.check_type(scores, np.ndarray)
    check_data.check_type(labels, np.ndarray)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            "The number of scores must be equal to the number of label subjects")


def _sort_list(list):
    indexes = np.argsort([np.median(x) for x in list])
    return indexes


def _process_scores_and_labels(scores, labels, sorted=False):
    _check_scores(scores, labels)
    scores_groups = []
    labels_unique = np.unique(labels)
    for label in labels_unique:
        res_score = scores[labels == label]
        scores_groups.append(res_score)

    if sorted:
        sorted_indices = _sort_list(scores_groups)
        scores_groups = [scores_groups[i] for i in sorted_indices]
        labels = labels[sorted_indices]
    else:
        sorted_indices = np.arange(len(labels))

    return scores_groups, labels, labels_unique, sorted_indices


def bar_plot(scores: np.ndarray, labels: np.ndarray,
             figsize: tuple = (10, 4), sorted: bool = False) -> Figure:
    """
    Creates a bar plot of subject scores.

    Parameters
    ----------
    scores : array-like of shape (n_subjects,)
        Array of subject scores.
    labels : array-like of shape (n_subjects,)
        Array of labels for each subject.
    figsize : tuple of two ints, optional
        Size of the figure in inches. Default is (10, 4).
    sorted : bool, optional
        If True, the bars are sorted by their values in ascending order. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The bar plot figure.
    """

    _check_scores(scores, labels)
    unique_labels, idx = np.unique(labels, return_index=True)
    unique_labels = unique_labels[np.argsort(idx)]
    cmap = colormap.discrete_colormap_nice(
        len(unique_labels), insert_white=False)
    colors = np.zeros((len(labels), 3))
    for label, color in zip(unique_labels, cmap.colors):
        colors[np.array(labels) == label] = color

    if sorted:
        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]
        colors = colors[sorted_indices]

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.hlines(y=0, xmin=-0.5, xmax=len(labels) - 0.5, colors='black')
    bars = ax.bar(np.arange(len(labels)), scores, color=colors)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Subject score')

    # Create legend with handles
    handles = [plt.Line2D([0], [0], color=cmap.colors[i], lw=4)
               for i in range(len(unique_labels))]
    ax.legend(handles, [f'{label}' for label in unique_labels], title='Groups')

    return fig


def violin_plot(scores: np.ndarray, labels: np.ndarray,
                figsize: tuple = (10, 4), sorted: bool = False) -> Figure:
    """
    Creates a violin plot of subject scores grouped by labels.

    Parameters
    ----------
    scores : np.ndarray
        Array of subject scores with shape (n_subjects,).
    labels : np.ndarray
        Array of labels for each subject with shape (n_subjects,).
    figsize : tuple of two ints, optional
        Size of the figure in inches. Default is (10, 4).
    sorted : bool, optional
        If True, the groups are sorted by their median values. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The violin plot figure.
    """

    _check_scores(scores, labels)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    scores_groups, labels, labels_unique, indices = _process_scores_and_labels(
        scores, labels, sorted)

    parts = ax.violinplot(dataset=scores_groups, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('red')
    ax.set_xticks(np.arange(1, len(labels_unique) + 1))
    ax.set_xticklabels(labels_unique)
    ax.set_ylabel('Subject score')
    return fig


def histogram_plot(scores: np.ndarray, labels: np.ndarray,
                   figsize: tuple = (10, 4), sorted: bool = False) -> Figure:
    """
    Creates a histogram plot with density estimation for subject scores grouped by labels.

    Parameters
    ----------
    scores : np.ndarray
        Array of subject scores with shape (n_subjects,).
    labels : np.ndarray
        Array of labels for each subject with shape (n_subjects,).
    figsize : tuple of two ints, optional
        Size of the figure in inches. Default is (10, 4).
    sorted : bool, optional
        If True, the groups are sorted by their median values.
        This ensures consistency with other plot functions. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The histogram plot figure.
    """

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    scores_groups, labels, labels_unique, indices = _process_scores_and_labels(
        scores, labels, sorted)

    left = np.min([np.min(x) for x in scores_groups]) - 0.5
    right = np.max([np.max(x) for x in scores_groups]) + 0.5

    bins_num = np.linspace(left, right, 20)

    for i, res_score in enumerate(scores_groups):
        den = stats.gaussian_kde(res_score)
        x = np.linspace(left, right, 100)
        ax.hist(res_score,
                bins=bins_num,
                label=labels_unique[indices[i]],
                alpha=0.5,
                edgecolor='black',
                linewidth=1.2,
                density=True)
        ax.plot(x, den(x), color='black')
    ax.set_xlabel('Subject score')
    ax.set_ylabel('Density')
    ax.legend()

    return fig


def get_plot(plot_type: str, scores: np.ndarray, labels: np.ndarray,
             figsize: tuple = (10, 4), sorted: bool = False):
    plot_functions = {
        "histogram": histogram_plot,
        "hist": histogram_plot,
        "bar": bar_plot,
        "barplot": bar_plot,
        "violin": violin_plot,
        "violinplot": violin_plot
    }
    if plot_type not in plot_functions:
        raise ValueError(
            f"Invalid plot type: {plot_type}. Expected one of {list(plot_functions.keys())}")

    plot_fig = plot_functions[plot_type](
        scores, labels, figsize=figsize, sorted=sorted)
    return plot_fig
