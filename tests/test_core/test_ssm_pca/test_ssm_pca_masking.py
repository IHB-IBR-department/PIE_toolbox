import numpy as np
from pie_toolbox.core.ssm_pca import masking


def test_get_threshold_mask_basic():
    m = np.array([[1., 2., 3.],
                  [3., 2., 1.]])
    mask = masking.get_threshold_mask(m, threshold=0.5)
    assert mask.shape == (3,)
    assert mask.dtype == bool


def test_get_loaded_mask_absolute():
    arr = np.array([0.1, 0.5, 0.9])
    mask = masking.get_loaded_mask(arr, threshold=0.5, is_threshold_relative=False)
    np.testing.assert_array_equal(mask, np.array([False, True, True]))


def test_get_loaded_mask_relative():
    arr = np.array([0.1, 0.5, 1.0])
    mask = masking.get_loaded_mask(arr, threshold=0.5, is_threshold_relative=True)
    np.testing.assert_array_equal(mask, np.array([False, True, True]))


def test_get_atlas_mask_basic():
    atlas = np.array([1, 2, 3, 4])
    mask = masking.get_atlas_mask(atlas, indexes=[2, 4])
    np.testing.assert_array_equal(mask, np.array([False, True, False, True]))


def test_mask_image_combined():
    m = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    mask_loaded = np.array([1., 0., 1.])
    atlas_loaded = np.array([1, 2, 3])
    voxels, mask = masking.mask_image(
        m,
        threshold=0.5,
        mask_loaded=mask_loaded,
        mask_loaded_threshold=0.5,
        mask_loaded_is_threshold_relative=False,
        atlas_loaded=atlas_loaded,
        indexes_mask=[1, 3]
    )
    assert voxels.shape[1] <= m.shape[1]
    assert mask.shape == (3,)
    assert mask.dtype == bool
