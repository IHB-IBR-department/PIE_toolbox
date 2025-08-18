import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from pie_toolbox.core.classifier import cross_val


class TestCrossVal(unittest.TestCase):

    def setUp(self):
        self.scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        self.labels = np.array([0, 0, 1, 1])
        self.predicted_labels = np.array([1])
        self.predicted_proba = np.array([[0.2, 0.8]])

    def test_CrossValResults_init_and_get_metrics(self):
        """Test CrossValResults object creation and get_metrics method."""
        predicted_labels = np.array([0, 1])
        predicted_scores = np.array([[0.8, 0.2], [0.1, 0.9]])
        
        with patch('pie_toolbox.core.classifier.metrics.Metrics') as mock_metrics:
            mock_metrics_instance = mock_metrics.return_value
            result = cross_val.CrossValResults(self.labels, predicted_labels, predicted_scores)
            result.get_metrics()
            
            mock_metrics.assert_called_once_with(self.labels, predicted_labels, predicted_scores)
            self.assertEqual(result.metrics, mock_metrics_instance)

    @patch('pie_toolbox.core.classifier.svm.SVM_Model')
    @patch('pie_toolbox.core.classifier.utils.fix_bad_groups')
    def test_LOO(self, mock_fix_bad_groups, mock_svm_model):
        """Test Leave-One-Out cross-validation."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = [np.array([0]), np.array([0]), np.array([1]), np.array([1])]
        mock_model_instance.predict_proba.side_effect = [np.array([[0.8, 0.2]]), np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]]), np.array([[0.2, 0.8]])]
        mock_svm_model.return_value = mock_model_instance
        mock_fix_bad_groups.side_effect = [np.array([[0.8, 0.2]]), np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]]), np.array([[0.2, 0.8]])]
        
        mock_split_iter = [(np.array([1, 2, 3]), np.array([0])),
                           (np.array([0, 2, 3]), np.array([1])),
                           (np.array([0, 1, 3]), np.array([2])),
                           (np.array([0, 1, 2]), np.array([3]))]

        with patch('sklearn.model_selection.LeaveOneOut.split', return_value=mock_split_iter) as mock_split:
            results = cross_val.LOO(self.scores, self.labels)
            
            self.assertEqual(len(results), 1)
            result = results[0]
            
            np.testing.assert_array_equal(result.real_labels, self.labels)
            
            expected_predicted_labels = np.array([0, 0, 1, 1]) 
            np.testing.assert_array_equal(result.predicted_labels, expected_predicted_labels)

            expected_predicted_scores = np.array([[0.8, 0.2], [0.9, 0.1], [0.1, 0.9], [0.2, 0.8]])
            np.testing.assert_array_equal(result.predicted_scores, expected_predicted_scores)
            
            mock_split.assert_called_once_with(self.scores, self.labels)
            self.assertEqual(mock_svm_model.call_count, 4)
            self.assertEqual(mock_model_instance.predict.call_count, 4)
            self.assertEqual(mock_model_instance.predict_proba.call_count, 4)
            self.assertEqual(mock_fix_bad_groups.call_count, 4)


    @patch('pie_toolbox.core.classifier.svm.SVM_Model')
    def test_Folds(self, mock_svm_model):
        """Test Stratified K-Fold cross-validation."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = [np.array([0, 1]), np.array([0, 1])]
        mock_model_instance.predict_proba.side_effect = [np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([[0.7, 0.3], [0.3, 0.7]])]
        mock_svm_model.return_value = mock_model_instance

        mock_split_iter = [(np.array([0, 2]), np.array([1, 3])),
                           (np.array([1, 3]), np.array([0, 2]))]
        
        with patch('sklearn.model_selection.StratifiedKFold.split', return_value=mock_split_iter) as mock_split:
            results = cross_val.Folds(self.scores, self.labels, folds=2)
            
            self.assertEqual(len(results), 2)

            result1 = results[0]
            np.testing.assert_array_equal(result1.real_labels, np.array([0, 1]))
            np.testing.assert_array_equal(result1.predicted_labels, np.array([0, 1]))
            np.testing.assert_array_almost_equal(result1.predicted_scores, np.array([[0.8, 0.2], [0.1, 0.9]]))

            result2 = results[1]
            np.testing.assert_array_equal(result2.real_labels, np.array([0, 1]))
            np.testing.assert_array_equal(result2.predicted_labels, np.array([0, 1]))
            np.testing.assert_array_almost_equal(result2.predicted_scores, np.array([[0.7, 0.3], [0.3, 0.7]]))
            
            mock_split.assert_called_once()
            self.assertEqual(mock_svm_model.call_count, 2)
            self.assertEqual(mock_model_instance.predict.call_count, 2)
            self.assertEqual(mock_model_instance.predict_proba.call_count, 2)

    def test_cross_val_from_model(self):
        """Test cross_val_from_model routing to correct functions."""
        mock_model = MagicMock()
        mock_model.scores_fitted = self.scores
        mock_model.labels_fitted = self.labels
        
        with patch('pie_toolbox.core.classifier.cross_val.LOO') as mock_loo:
            cross_val.cross_val_from_model(mock_model, split_type='LOO')
            mock_loo.assert_called_once_with(self.scores, self.labels)

        with patch('pie_toolbox.core.classifier.cross_val.Folds') as mock_folds:
            cross_val.cross_val_from_model(mock_model, split_type='Folds', folds=3)
            mock_folds.assert_called_once_with(self.scores, self.labels, folds=3)
            
        with self.assertRaises(ValueError):
            cross_val.cross_val_from_model(mock_model, split_type='Invalid')

if __name__ == '__main__':
    unittest.main()