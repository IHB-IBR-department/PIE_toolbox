import numpy as np
from pie_toolbox.core.ssm_pca import pca


def test_ssm_pca_shapes_and_sorting():
    m = np.array([[1., 2.],
                  [3., 4.]])
    patterns, eigvals = pca.ssm_pca(m)
    assert patterns.shape == m.shape
    assert eigvals.shape[0] == m.shape[0]
    assert np.all(eigvals[:-1] >= eigvals[1:])
