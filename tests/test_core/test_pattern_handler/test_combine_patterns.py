import unittest
import numpy as np
from pie_toolbox.core.pattern_handler.combine_patterns import combine_patterns, logreg_pattern_coefficients

class TestCombinePatterns(unittest.TestCase):

    def test_combine_patterns_equal_weights(self):
        """Test that combine_patterns sums patterns when no coefficients are given."""
        patterns = np.array([[1, 2], [3, 4]])
        combined = combine_patterns(patterns)
        expected = np.array([4, 6])
        np.testing.assert_array_equal(combined, expected)

    def test_combine_patterns_with_coefficients(self):
        """Test that combine_patterns correctly applies given coefficients."""
        patterns = np.array([[1, 2], [3, 4]])
        coeffs = np.array([2, 3])
        combined = combine_patterns(patterns, coeffs)
        expected = np.array([11, 16])
        np.testing.assert_array_equal(combined, expected)

    def test_logreg_pattern_coefficients_shape(self):
        """Test that logreg_pattern_coefficients returns coefficients of correct shape and AIC as float."""
        scores = np.array([[0.1, 0.2], [0.3, 0.4]])
        labels = np.array([0, 1])
        coefs, aic = logreg_pattern_coefficients(scores, labels)
        self.assertEqual(coefs.shape, (2,))
        self.assertIsInstance(aic, float)


if __name__ == '__main__':
    unittest.main()
