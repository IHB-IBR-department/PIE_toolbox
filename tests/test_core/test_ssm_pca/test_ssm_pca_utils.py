import numpy as np
from pie_toolbox.core.ssm_pca import utils


def test_get_sub_by_sub_covmat_shape_and_values():
    m = np.array([[1., 2.],
                  [3., 4.]])
    cov = utils.get_sub_by_sub_covmat(m)
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov, m @ m.T)


def test_compute_sorted_eigenpairs_symmetry():
    m = np.array([[2., 1.],
                  [1., 2.]])
    eigvals, eigvecs = utils.compute_sorted_eigenpairs(m)
    assert np.all(eigvals[:-1] >= eigvals[1:])
    assert eigvecs.shape == (2, 2)


def test_weight_eigenvectors_by_eigenvalues_scaling():
    eigvecs = np.array([[1., 0.],
                        [0., 1.]])
    eigvals = np.array([4., 9.])
    weighted = utils.weight_eigenvectors_by_eigenvalues(eigvecs, eigvals)
    expected = np.array([[2., 0.],
                         [0., 3.]])
    np.testing.assert_allclose(weighted, expected)


def test_subtract_mean_no_mask():
    arr = np.array([1., 2., 3.])
    res = utils.subtract_mean(arr)
    assert np.isclose(np.mean(res), 0.0)


def test_subtract_mean_with_mask():
    arr = np.array([1., 2., 3.])
    mask = np.array([True, False, True])
    res = utils.subtract_mean(arr, mask=mask)
    expected_mean = np.mean(arr[mask])
    np.testing.assert_allclose(res, arr - expected_mean)


def test_get_explained_variance_sum_to_one():
    eigvals = np.array([2., 2., 6.])
    var = utils.get_explained_variance(eigvals)
    np.testing.assert_allclose(np.sum(var), 1.0)


def test_invert_patterns_binary_labels():
    patterns = np.array([[1., -1.], [2., -2.]])
    scores = np.array([[1., 1.], [2., 2.]])
    labels = np.array([0, 1])
    p2, s2 = utils.invert_patterns(patterns.copy(), scores.copy(), labels)
    assert p2.shape == patterns.shape
    assert s2.shape == scores.shape
