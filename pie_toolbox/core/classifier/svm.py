import numpy as np
from sklearn import svm


class SVM_Model:
    def __init__(self, scores: np.ndarray, labels: np.ndarray, kernel: str = 'rbf', regularization_param: float = 1):  # 1d arrays
        """
        Initialize the SVM_Model with given scores, labels, kernel type, and regularization parameter.

        Parameters
        ----------
        scores : np.ndarray
            2D array representing the feature scores for training the SVM.
        labels : np.ndarray
            1D array containing the class labels corresponding to the scores.
        kernel : str
            The kernel type to be used in the SVM algorithm.
        regularization_param : float
            The regularization parameter C for the SVM model.
        """

        self.model = svm.SVC(
            kernel=kernel,
            C=regularization_param,
            probability=True,
            class_weight='balanced')
        self.model.fit(scores, labels)
        self.scores_fitted = scores
        self.labels_fitted = labels

    def predict(self, scores: np.ndarray):
        self.labels_predicted = self.model.predict(scores)
        return self.labels_predicted

    def predict_proba(self, scores: np.ndarray):
        self.scores_predicted = self.model.predict_proba(scores)
        return self.scores_predicted
