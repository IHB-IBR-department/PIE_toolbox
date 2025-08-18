from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np
from pie_toolbox.core.classifier import utils
from pie_toolbox.core.validators import check_labels, check_data


class Metrics:

    def __init__(self, real_labels, predicted_labels, predicted_scores):
        """
        Initialize the Metrics object with the given real labels, predicted labels, and predicted scores.

        Parameters
        ----------
        real_labels : array-like of shape (n_samples,)
            Real labels of the samples.
        predicted_labels : array-like of shape (n_samples,)
            Predicted labels of the samples.
        predicted_scores : array-like of shape (n_samples,)
            Predicted scores or probabilities for the positive class.

        Attributes
        ----------
        n_subjects : array-like of shape (n_classes,)
            Number of subjects for each class.
        confusion_matrix : array-like of shape (n_classes, n_classes)
            Confusion matrix.
        true_pos : array-like of shape (n_classes,)
            Number of true positives for each class.
        false_pos : array-like of shape (n_classes,)
            Number of false positives for each class.
        false_neg : array-like of shape (n_classes,)
            Number of false negatives for each class.
        true_neg : array-like of shape (n_classes,)
            Number of true negatives for each class.
        total : int
            Total number of samples.
        sensitivity : array-like of shape (n_classes,)
            Sensitivity of the model for each class.
        recall : array-like of shape (n_classes,)
            Recall of the model for each class.
        specificity : array-like of shape (n_classes,)
            Specificity of the model for each class.
        accuracy : array-like of shape (n_classes,)
            Accuracy of the model for each class.
        balanced_accuracy : array-like of shape (n_classes,)
            Balanced accuracy of the model for each class.
        ppv : array-like of shape (n_classes,)
            Positive predictive value of the model for each class.
        precision : array-like of shape (n_classes,)
            Precision of the model for each class.
        npv : array-like of shape (n_classes,)
            Negative predictive value of the model for each class.
        accuracy_global : float
            Global accuracy of the model.

        Returns
        -------
        Metrics
            Metrics object with the calculated metrics.
        """
        check_labels.check_labels(real_labels)
        check_data.check_type(predicted_scores, np.ndarray)

        self.real_labels = real_labels
        self.predicted_labels = predicted_labels
        self.predicted_scores = predicted_scores

        # Get confusion matrix
        self.confusion_matrix = self.get_cm()

        self.n_subjects = np.sum(self.confusion_matrix, axis=1)
        self.true_pos = np.diag(self.confusion_matrix)
        self.false_pos = np.sum(self.confusion_matrix, axis=0) - self.true_pos
        self.false_neg = np.sum(self.confusion_matrix, axis=1) - self.true_pos
        self.true_neg = self.confusion_matrix.sum(
            axis=(0, 1)) - self.true_pos - self.false_pos - self.false_neg
        self.total = np.sum(self.true_pos) + np.sum(self.true_neg) + \
            np.sum(self.false_pos) + np.sum(self.false_neg)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.sensitivity = self.true_pos / \
                (self.true_pos + self.false_neg)  # recall
            self.recall = self.sensitivity
            self.specificity = self.true_neg / (self.true_neg + self.false_pos)
            self.accuracy = (self.true_pos + self.true_neg) / \
                (self.true_pos + self.true_neg + self.false_pos + self.false_neg)
            self.balanced_accuracy = (self.sensitivity + self.specificity) / 2
            self.ppv = self.true_pos / \
                (self.true_pos + self.false_pos)  # precision
            self.precision = self.ppv
            self.npv = self.true_neg / (self.true_neg + self.false_neg)
            self.true_pos = self.true_pos
            self.false_pos = self.false_pos
            self.true_neg = self.true_neg
            self.false_neg = self.false_neg
            self.accuracy_global = (
                np.sum(self.true_pos) + np.sum(self.true_neg)) / self.total

        # Get AUC
        # self.get_auc() returns the AUC, false positive rates, and true positive rates
        # We just want the AUC for each class
        self.auc = np.array([self.get_auc(class_name=label)[0]
                            for label in np.unique(self.real_labels)])

    def get_cm(self):
        """
        Get confusion matrix.

        Parameters
        ----------
        real_labels : array-like of shape (n_samples,)
            True labels of the samples.
        predicted_labels : array-like of shape (n_samples,)
            Predicted labels of the samples.

        Returns
        -------
        cm : array-like of shape (n_classes, n_classes)
            Confusion matrix.
        """

        return confusion_matrix(self.real_labels, self.predicted_labels)

    def get_auc(self, class_name=None):
        """
        Get AUC.

        Parameters
        ----------
        class_name : str, optional
            Class name for which to compute the AUC.

        Returns
        -------
        roc_auc : float
            Area under the ROC curve.
        fpr : array-like of shape (n_samples,)
            False positive rates.
        tpr : array-like of shape (n_samples,)
            True positive rates.
        """
        if class_name is None:
            class_name = np.unique(self.real_labels)[1]
        roc_auc, fpr, tpr = utils.get_auc(
            self.real_labels, self.predicted_scores, class_name=class_name)
        return roc_auc, fpr, tpr
