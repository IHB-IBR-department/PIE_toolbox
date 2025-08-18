from pie_toolbox.workflows.image_dataset import ImageDataset
from pie_toolbox.workflows.ssm_pca import VoxelPCA
from pie_toolbox.core.classifier import svm, cross_val
from pie_toolbox.core.classifier import metrics as classifier_metrics
from pie_toolbox.core.validators import check_data
from pie_toolbox.core.common import logger
from pie_toolbox.core.common import converters

import numpy as np


class SVM_Classifier():
    model: svm.SVM_Model = None
    metrics: classifier_metrics.Metrics = None

    def __init__(self):
        pass

    def fit(self, scores: np.ndarray | list, labels: np.ndarray,
            kernel: str = 'rbf', regularization_param: float = 1.0):
        if isinstance(scores, list):
            if all(isinstance(score, np.ndarray) for score in scores):
                scores = converters.get_concatenated_scores(scores)
            else:
                raise ValueError("All score arrays must be numpy arrays")
        check_data.check_type(scores, np.ndarray)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        check_data.check_dimensions_number(scores, 2)
        check_data.check_type(labels, np.ndarray)
        self.model = svm.SVM_Model(
            scores=scores,
            labels=labels,
            kernel=kernel,
            regularization_param=regularization_param)

    def cross_validation(self, split_type='LOO', folds=5):
        """
        Perform cross-validation on the model with the given parameters.

        Parameters
        ----------
        split_type : str, optional
            The type of cross-validation to use. Options are 'LOO' for Leave-One-Out
            or 'Folds' for StratifiedKFold. Default is 'LOO'.
        folds : int, optional
            The number of folds for StratifiedKFold. Default is 5.

        Updates
        -------
        self.metrics : Metrics
            A list of CrossValResults objects, each containing evaluation metrics from a LOO or single fold in the cross-validation.
        """

        if self.model is None:
            raise ValueError("Model is not initialized")
        self.metrics = cross_val.cross_val_from_model(
            self.model, split_type=split_type, folds=folds)

    def predict(self, scores: np.ndarray):
        """
        Predict labels for given feature scores.

        Parameters
        ----------
        scores : np.ndarray
            2D array representing the feature scores to be predicted.

        Returns
        -------
        np.ndarray
            1D array of predicted labels
        """

        return self.model.predict(scores)

    def predict_proba(self, scores: np.ndarray):
        """
        Predict probability scores for given feature scores.

        Parameters
        ----------
        scores : np.ndarray
            2D array representing the feature scores to be predicted.

        Returns
        -------
        np.ndarray
            2D array of predicted probability scores, with shape (n_samples, n_classes)
        """
        return self.model.predict_proba(scores)

    def get_metrics(self, real_labels, predicted_labels, predicted_scores):
        """
        Get the metrics for the given real labels, predicted labels and predicted scores.

        Parameters
        ----------
        real_labels : np.ndarray
            1D array representing the real labels of the samples.
        predicted_labels : np.ndarray
            1D array representing the predicted labels of the samples.
        predicted_scores : np.ndarray
            2D array of predicted probability scores, with shape (n_samples, n_classes)

        Returns
        -------
        Metrics
            Object containing the metrics.
        """
        self.metrics = classifier_metrics.Metrics(
            real_labels, predicted_labels, predicted_scores)
        return self.metrics

    def validation(self, scores: np.ndarray,
                   labels: np.ndarray) -> classifier_metrics.Metrics:
        """
        Perform validation of the model on the given scores and labels.

        Parameters
        ----------
        scores : np.ndarray
            2D array representing the feature scores to be validated.
        labels : np.ndarray
            1D array representing the real labels of the samples.

        Returns
        -------
        Metrics
            Object containing the metrics from the validation.
        """
        labels_predicted = self.predict(scores)
        probabilities_predicted = self.predict_proba(scores)

        metrics_validation = self.get_metrics(
            real_labels=labels,
            predicted_labels=labels_predicted,
            predicted_scores=probabilities_predicted)

        return metrics_validation
