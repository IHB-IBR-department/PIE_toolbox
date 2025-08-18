import unittest
import numpy as np
from pie_toolbox.core.classifier.utils import get_random_state, get_cv_split, get_auc, fix_bad_groups

class TestUtils(unittest.TestCase):

    def test_get_random_state_type(self):
        """Test that get_random_state returns an integer within expected range."""
        state = get_random_state()
        self.assertIsInstance(state, int)
        self.assertGreaterEqual(state, 1)
        self.assertLess(state, 999)

    def test_get_cv_split_loo(self):
        """Test Leave-One-Out CV split correctness."""
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0,1,0,1,0,1,0,1,0,1])
        splits, n_splits = get_cv_split(X, y, split_type="LOO")
        splits_list = list(splits)
        self.assertEqual(n_splits, 10)
        for train_idx, test_idx in splits_list:
            self.assertEqual(len(train_idx), 9)
            self.assertEqual(len(test_idx), 1)

    def test_get_cv_split_folds(self):
        """Test k-fold CV split correctness."""
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0,1,0,1,0,1,0,1,0,1])
        splits, n_splits = get_cv_split(X, y, split_type="Folds", folds=5, random_state=42)
        splits_list = list(splits)
        self.assertEqual(n_splits, 5)
        for train_idx, test_idx in splits_list:
            self.assertEqual(len(train_idx), 8)
            self.assertEqual(len(test_idx), 2)

    def test_get_auc_basic(self):
        """Test that get_auc returns valid AUC and matching FPR/TPR shapes."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([[0.9, 0.1],
                             [0.8, 0.2],
                             [0.2, 0.8],
                             [0.1, 0.9]])
        auc_val, fpr, tpr = get_auc(y_true, y_scores)
        self.assertGreaterEqual(auc_val, 0)
        self.assertLessEqual(auc_val, 1)
        self.assertEqual(fpr.shape, tpr.shape)

    def test_fix_bad_groups_no_missing(self):
        """Test that fix_bad_groups returns unchanged array if no labels are missing."""
        pred = np.array([[0.5, 0.5]])
        test_labels = np.array(['cat'])
        train_labels = np.array(['cat', 'dog'])
        all_labels = np.array(['cat', 'dog'])
        fixed = fix_bad_groups(pred, test_labels, train_labels, all_labels)
        np.testing.assert_allclose(fixed, pred)

    def test_fix_bad_groups_with_missing(self):
        """Test that missing test labels are inserted with zero probability."""
        pred = np.array([[0.7, 0.1, 0.2]])
        test_labels = np.array(['dog'])
        train_labels = np.array(['cat', 'fish', 'bird'])
        all_labels = np.array(['cat', 'dog', 'fish', 'bird'])
        fixed = fix_bad_groups(pred, test_labels, train_labels, all_labels)
        expected = np.array([[0.7, 0.1, 0.0, 0.2]])
        np.testing.assert_allclose(fixed, expected)


if __name__ == '__main__':
    unittest.main()
