import unittest
import numpy as np
from unittest.mock import patch
from pie_toolbox.core.classifier.metrics import Metrics

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.real_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.predicted_labels = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 0])
        self.predicted_scores = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9],
                                          [0.8, 0.2], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
        
        self.expected_cm = np.array([[4, 1], [1, 4]])
        
        self.true_pos = np.diag(self.expected_cm)
        self.false_pos = np.sum(self.expected_cm, axis=0) - self.true_pos
        self.false_neg = np.sum(self.expected_cm, axis=1) - self.true_pos
        self.true_neg = self.expected_cm.sum() - self.true_pos - self.false_pos - self.false_neg
        self.global_accuracy = self.expected_cm.trace() / self.real_labels.size

    def test_metrics_init_and_calculations(self):
        """Test initialization and all metric calculations."""
        with patch('pie_toolbox.core.classifier.utils.get_auc', return_value=(0.8, np.array([0.1, 0.2]), np.array([0.9, 0.8]))):
            metrics_obj = Metrics(self.real_labels, self.predicted_labels, self.predicted_scores)

        np.testing.assert_array_equal(metrics_obj.confusion_matrix, self.expected_cm)
        
        np.testing.assert_array_equal(metrics_obj.true_pos, self.true_pos)
        np.testing.assert_array_equal(metrics_obj.false_pos, self.false_pos)
        np.testing.assert_array_equal(metrics_obj.false_neg, self.false_neg)
        np.testing.assert_array_equal(metrics_obj.true_neg, self.true_neg)

        with np.errstate(divide='ignore', invalid='ignore'):
            expected_sensitivity = self.true_pos / (self.true_pos + self.false_neg)
            expected_specificity = self.true_neg / (self.true_neg + self.false_pos)
            expected_precision = self.true_pos / (self.true_pos + self.false_pos)
            expected_npv = self.true_neg / (self.true_neg + self.false_neg)
            expected_balanced_accuracy = (expected_sensitivity + expected_specificity) / 2
        
        np.testing.assert_array_almost_equal(metrics_obj.sensitivity, expected_sensitivity)
        np.testing.assert_array_almost_equal(metrics_obj.specificity, expected_specificity)
        np.testing.assert_array_almost_equal(metrics_obj.precision, expected_precision)
        np.testing.assert_array_almost_equal(metrics_obj.npv, expected_npv)
        np.testing.assert_array_almost_equal(metrics_obj.balanced_accuracy, expected_balanced_accuracy)
        
        np.testing.assert_almost_equal(metrics_obj.accuracy_global, self.global_accuracy)
        
    @patch('pie_toolbox.core.classifier.utils.get_auc')
    def test_get_auc_method(self, mock_get_auc):
        """Test the get_auc method of the Metrics class."""
        mock_get_auc.return_value = (0.8, np.array([0.1, 0.2]), np.array([0.9, 0.8]))
        metrics_obj = Metrics(self.real_labels, self.predicted_labels, self.predicted_scores)
        auc_val, fpr, tpr = metrics_obj.get_auc(class_name=1)
        
        self.assertEqual(auc_val, 0.8)
        np.testing.assert_array_equal(fpr, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(tpr, np.array([0.9, 0.8]))
        
        mock_get_auc.assert_called_with(self.real_labels, self.predicted_scores, class_name=1)

if __name__ == '__main__':
    unittest.main()