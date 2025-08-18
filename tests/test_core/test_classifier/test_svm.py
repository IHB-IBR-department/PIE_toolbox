import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from pie_toolbox.core.classifier.svm import SVM_Model


class TestSVMModel(unittest.TestCase):

    def setUp(self):
        self.scores = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.labels = np.array([0, 0, 1, 1])

    @patch('sklearn.svm.SVC')
    def test_init(self, mock_svc):
        """Test SVM_Model initialization and fitting."""
        mock_model_instance = MagicMock()
        mock_svc.return_value = mock_model_instance
        
        model = SVM_Model(self.scores, self.labels, kernel='rbf', regularization_param=2)
        
        mock_svc.assert_called_once_with(kernel='rbf', C=2, probability=True, class_weight='balanced')
        mock_model_instance.fit.assert_called_once_with(self.scores, self.labels)
        
        np.testing.assert_array_equal(model.scores_fitted, self.scores)
        np.testing.assert_array_equal(model.labels_fitted, self.labels)
        self.assertEqual(model.model, mock_model_instance)

    @patch('sklearn.svm.SVC')
    def test_predict(self, mock_svc):
        """Test the predict method."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = np.array([1, 1])
        mock_svc.return_value = mock_model_instance
        
        model = SVM_Model(self.scores, self.labels)
        new_scores = np.array([[5, 6], [6, 7]])
        predictions = model.predict(new_scores)
        
        mock_model_instance.predict.assert_called_once_with(new_scores)
        np.testing.assert_array_equal(predictions, np.array([1, 1]))
        np.testing.assert_array_equal(model.labels_predicted, np.array([1, 1]))

    @patch('sklearn.svm.SVC')
    def test_predict_proba(self, mock_svc):
        """Test the predict_proba method."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict_proba.return_value = np.array([[0.2, 0.8], [0.1, 0.9]])
        mock_svc.return_value = mock_model_instance
        
        model = SVM_Model(self.scores, self.labels)
        new_scores = np.array([[5, 6], [6, 7]])
        probabilities = model.predict_proba(new_scores)
        
        mock_model_instance.predict_proba.assert_called_once_with(new_scores)
        np.testing.assert_array_almost_equal(probabilities, np.array([[0.2, 0.8], [0.1, 0.9]]))
        np.testing.assert_array_almost_equal(model.scores_predicted, np.array([[0.2, 0.8], [0.1, 0.9]]))

if __name__ == '__main__':
    unittest.main()