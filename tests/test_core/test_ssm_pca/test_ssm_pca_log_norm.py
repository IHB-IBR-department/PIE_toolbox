import numpy as np
from pie_toolbox.core.ssm_pca import log_norm


def test_log_voxels_basic():
    m = np.array([[1.0, 0.0, -1.0, 4.0]])
    result = log_norm.log_voxels(m.copy())
    expected = np.array([[np.log(1.0), 0.0, 0.0, np.log(4.0)]])
    np.testing.assert_allclose(result, expected)


def test_normalize_no_mask():
    m = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    normed, gmp = log_norm.normalize(m.copy())
    np.testing.assert_allclose(np.mean(normed, axis=0), 0.0)
    np.testing.assert_allclose(np.mean(normed, axis=1), 0.0)
    assert gmp.shape == (m.shape[1],)


def test_normalize_with_mask():
    m = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    mask = np.array([True, False, True])
    normed, gmp = log_norm.normalize(m.copy(), seed_mask=mask)
    np.testing.assert_allclose(np.mean(normed, axis=0), 0.0)
    assert gmp.shape == (3,)


def test_log_normalize_with_and_without_log():
    m = np.array([[1., 2.], [3., 4.]])
    norm_log, gmp_log = log_norm.log_normalize(m.copy(), log_transform=True)
    norm_no_log, gmp_no_log = log_norm.log_normalize(m.copy(), log_transform=False)
    assert norm_log.shape == m.shape
    assert norm_no_log.shape == m.shape
    assert gmp_log.shape == (m.shape[1],)
    assert gmp_no_log.shape == (m.shape[1],)

def test_subtract_gmp_basic():
    data = np.array([[1., 2., 3.],
                     [4., 5., 6.]])
    gmp = np.array([0.5, 1.5, 2.5])
    mask = np.array([True, True, True])

    expected = []
    for subj in data:
        subj_centered = subj - subj[mask].mean()
        expected.append(subj_centered - gmp)
    expected = np.array(expected)

    result = log_norm.subtract_gmp(data.copy(), gmp, mask)

    assert result.shape == data.shape
    np.testing.assert_allclose(result, expected, atol=1e-8)