from matplotlib import colors as mcolors
import numpy as np


def discrete_colormap(n: int, insert_white: bool = True):
    """
    Creates a discrete colormap with n colors

    Parameters
    ----------
    n : int
        The number of colors in the colormap
    insert_white : bool, optional
        Whether to insert white color at the beginning of the colormap, by default True
    Returns
    -------
    matplotlib.colors.ListedColormap
        The created colormap
    """
    if not insert_white:
        n += 1
    if (n > 2):
        hues = np.linspace(0, 1, n)[:-1]
    else:
        hues = np.linspace(1, 1, 3)
    # Convert hues to RGB colors using hsv
    rgb_colors = [mcolors.hsv_to_rgb((hue, 1, 1)) for hue in hues]
    if (insert_white):
        rgb_colors.insert(0, (1, 1, 1))
    # Create a ListedColormap
    cmap = mcolors.ListedColormap(rgb_colors)
    return cmap


def discrete_colormap_nice(n: int, insert_white: bool = True):

    # Limit the number of colors to a maximum of 12
    if n > 12:
        return discrete_colormap(n, insert_white)

    if not insert_white:
        n += 1

    # Predefined base colors
    base_colors = [
        (0.95, 0.35, 0.4), (0.4, 0.7, 0.8), (0.85, 0.65, 0.4), (0.7, 0.35, 0.8),
        (0.2, 0.8, 0.95), (0.9, 0.7, 0.4), (0.3, 0.9, 0.7), (0.8, 0.9, 0.4),
        (0.5, 0.7, 0.8), (0.9, 0.5, 0.7), (0.7, 0.9, 0.5), (0.4, 0.7, 0.7),
        (0.7, 0.7, 0.7)
    ][:n]

    if insert_white:
        base_colors.insert(0, (1, 1, 1))

    # Create a ListedColormap
    cmap = mcolors.ListedColormap(base_colors)
    return cmap
