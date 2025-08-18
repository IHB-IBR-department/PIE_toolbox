from pie_toolbox.core.classifier import utils, svm, metrics
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import numpy as np
from pie_toolbox.core.common.logger import setup_root_logger, get_logger

setup_root_logger('cross_val')
logger = get_logger('cross_val')


class CrossValResults():
    def __init__(self, real_labels, predicted_labels,
                 predicted_scores, metrics=None):
        self.real_labels = real_labels
        self.predicted_labels = predicted_labels
        self.predicted_scores = predicted_scores
        self.metrics = metrics

    def get_metrics(self):
        self.metrics = metrics.Metrics(
            self.real_labels,
            self.predicted_labels,
            self.predicted_scores)
        return self.metrics


def LOO(scores: np.ndarray, labels: np.ndarray, kernel='rbf'):
    """
    Leave-One-Out cross-validation for the given scores and labels.

    Parameters
    ----------
    scores : np.ndarray
        1D array representing the feature scores for training the SVM.
    labels : np.ndarray
        1D array containing the class labels corresponding to the scores.
    kernel : str
        The kernel type to be used in the SVM algorithm.

    Returns
    -------
    list
        A list containing a single CrossValResults object with the metrics and predictions.
    """

    cv_split = LeaveOneOut().split(scores, labels)
    predicted_labels = np.array([])
    predicted_scores = np.array([])

    for train, test in cv_split:
        train_labels, _ = np.unique(labels[train], return_counts=True)
        test_labels, _ = np.unique(labels[test], return_counts=True)

        cross_val_model = svm.SVM_Model(
            scores[train], labels[train], kernel=kernel)

        predict_label = cross_val_model.predict(scores[test])
        predict_proba_score = cross_val_model.predict_proba(scores[test])

        predict_proba_score = utils.fix_bad_groups(
            predict_proba_score, test_labels, train_labels, labels)

        # Save real and predicted labels and scores
        predicted_labels = np.concatenate((predicted_labels, predict_label))
        predicted_scores = np.vstack(
            (predicted_scores,
             predict_proba_score)) if predicted_scores.size else predict_proba_score

    result = CrossValResults(labels, predicted_labels, predicted_scores)
    result.get_metrics()

    return [result]


def Folds(scores: np.ndarray, labels: np.ndarray, kernel='rbf', folds=5):
    """
    Perform Stratified K-Fold cross-validation on the provided scores and labels
    using an SVM model.

    Parameters
    ----------
    scores : np.ndarray
        1D array representing the feature scores for training the SVM.
    labels : np.ndarray
        1D array containing the class labels corresponding to the scores.
    kernel : str, optional
        The kernel type to be used in the SVM algorithm. Default is 'rbf'.
    folds : int, optional
        The number of folds for Stratified K-Fold cross-validation. Default is 5.

    Returns
    -------
    list
        A list of CrossValResults objects containing the metrics and predictions
        for each fold.

    Notes
    -----
    This function will skip any fold where a label from the test set does not
    appear in the training set.
    """

    cv_split = StratifiedKFold(
        n_splits=folds,
        shuffle=True).split(
        scores,
        labels)

    result = []

    for train, test in cv_split:
        train_labels, _ = np.unique(labels[train], return_counts=True)
        test_labels, _ = np.unique(labels[test], return_counts=True)

        cross_val_model = svm.SVM_Model(
            scores[train], labels[train], kernel=kernel)

        predict_labels = cross_val_model.predict(scores[test])
        predict_proba_scores = cross_val_model.predict_proba(scores[test])

        # Skip the parts when label from test does not appear in train
        if not np.isin(test_labels, train_labels).all():
            logger.warning(
                f"Skipping fold since a label from test ({test_labels}) does not appear in train ({train_labels})")
            continue

        result_intermediate = CrossValResults(
            labels[test], predict_labels, predict_proba_scores)
        result_intermediate.get_metrics()
        result.append(result_intermediate)

    return result


def cross_val_from_model(model: svm.SVM_Model,
                         split_type='LOO', folds=5) -> list:
    """
    Perform cross-validation for a given SVM model.

    Parameters
    ----------
    model : SVM_Model
        The SVM model to evaluate.
    split_type : str, optional
        The type of cross-validation to use. Options are 'LOO' for Leave-One-Out
        or 'Folds' for StratifiedKFold. Default is 'LOO'.
    folds : int, optional
        The number of folds for StratifiedKFold. Default is 5.

    Returns
    -------
    result : list
        A list of CrossValResults objects, each containing evaluation metrics from a LOO or single fold in the cross-validation.
    """

    if split_type == 'LOO':
        result = LOO(model.scores_fitted, model.labels_fitted)
    elif split_type == 'Folds':
        result = Folds(model.scores_fitted, model.labels_fitted, folds=folds)
    else:
        raise ValueError("Invalid split_type. Expected 'LOO' or 'Folds'.")

    return result
