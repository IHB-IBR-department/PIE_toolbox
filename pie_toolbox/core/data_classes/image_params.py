from dataclasses import dataclass
import numpy as np


@dataclass
class ImageParams():
    # IMAGE PROPERTIES
    shape: tuple = None
    affine: np.ndarray = None
    zeros_mask: np.ndarray = None
    gmp: np.ndarray = None
    seed_mask: np.ndarray = None
