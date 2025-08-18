import unittest
import numpy as np
from pie_toolbox.core.pattern_handler.scores import get_scores, get_single_subject_scores

class TestScores(unittest.TestCase):

    def test_get_single_subject_scores_whole_brain(self):
        """Test get_single_subject_scores for whole-brain pattern (atlas all ones)."""
        voxels = np.array([1, 2, 3])
        pattern = np.array([0.5, 0.5, 0.5])
        atlas = np.ones(3)
        scores = get_single_subject_scores(voxels, pattern, atlas)
        expected = np.array([3.0])
        np.testing.assert_allclose(scores, expected)

    def test_get_single_subject_scores_regions(self):
        """Test get_single_subject_scores when atlas has multiple regions."""
        voxels = np.array([1, 2, 3, 4])
        pattern = np.array([1, 1, 1, 1])
        atlas = np.array([1, 1, 2, 2])
        scores = get_single_subject_scores(voxels, pattern, atlas)
        expected = np.array([3, 7])
        np.testing.assert_allclose(scores, expected)

    def test_get_scores_output_shape(self):
        """Test get_scores returns a list of correct shapes."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        patterns = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        scores_list = get_scores(data, patterns)
        self.assertEqual(len(scores_list), 2)
        for arr in scores_list:
            self.assertEqual(arr.shape[0], data.shape[0])

if __name__ == '__main__':
    unittest.main()
