from pie_toolbox.core.validators import check_data, check_labels
from pie_toolbox.core.common import converters
import numpy as np
import scipy


def t_test(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Perform a two-sample t-test on the given scores.

    Parameters
    ----------
    scores : np.ndarray
        Array of scores with shape (n_subjects, n_regions).
    labels : np.ndarray
        1D array of labels with shape (n_subjects,).
        Must contain exactly two unique labels.

    Returns
    -------
    p_values : np.ndarray
        The p-values of the t-tests between the two groups of scores, with shape (n_regions,).

    """

    check_data.check_type(scores, np.ndarray)
    check_data.check_type(labels, np.ndarray)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            "The number of scores must be equal to the number of label subjects")
    unique_vals = np.unique(labels)
    if unique_vals.size != 2:
        raise ValueError("Labels array must have exactly two unique values")

    labels_binary = labels == unique_vals[0]
    scores_0 = scores[labels_binary]
    scores_1 = scores[~labels_binary]
    t_stat, p_val = scipy.stats.ttest_ind(scores_0, scores_1, axis=0)

    return p_val


def aic(log_likelihood: float, n_params: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for a given model.

    Parameters
    ----------
    log_likelihood : float
        The log likelihood of the model.
    n_params : int
        The number of parameters in the model.

    Returns
    -------
    aic : float
        The AIC value of the model.

    Notes
    -----
    The AIC is calculated as -2 * log_likelihood + 2 * n_params.
    """
    return -2 * log_likelihood + 2 * n_params


def log_likelihood(model: object, data: np.ndarray,
                   labels: np.ndarray) -> float:
    """
    Calculate the log likelihood of the data given the model.

    Parameters
    ----------
    model : object
        A fitted model object that has a `predict_proba` method to return predicted probabilities.
    data : np.ndarray
        2D array of input data for which the probabilities are predicted.
    labels : np.ndarray
        1D array of true class labels corresponding to the data samples.

    Returns
    -------
    log_likelihood : float
        The computed log likelihood value of the model given the data and labels.

    Notes
    -----
    This function assumes that the model is already fitted and can predict probabilities.
    The log likelihood is computed by summing the log of predicted probabilities for the true labels.
    """

    check_labels.check_labels(labels)
    predicted_prob = model.predict_proba(data)
    labels_indexes = converters.labels_to_indexes(labels)
    log_likelihood = np.sum(
        np.log(predicted_prob[np.arange(len(labels_indexes)), labels_indexes]))
    return log_likelihood
