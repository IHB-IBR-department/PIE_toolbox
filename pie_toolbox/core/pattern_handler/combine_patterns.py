from sklearn.linear_model import LogisticRegression
import numpy as np
from pie_toolbox.core.common import stats


def logreg_pattern_coefficients(scores: np.ndarray, labels: np.ndarray):
    """
    Calculate the coefficients and AIC of a logistic regression model
    for the given scores and labels.

    Parameters
    ----------
    scores : array-like
        2D array of feature scores (n_subjects, n_patterns).
    labels : array-like
        1D array of class labels for each sample (n_subjects).

    Returns
    -------
    coefficients : array-like
        1D array of coefficients for each feature (n_patterns).
    aic : float
        Akaike information criterion of the model.
    """
    with np.errstate(all='ignore'):
        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(scores, labels)
    log_likelihood = stats.log_likelihood(logreg, scores, labels)
    aic = stats.aic(log_likelihood, logreg.coef_.shape[0])
    return logreg.coef_[0], aic


def combine_patterns(patterns: np.ndarray,
                     coefficients: np.ndarray = None) -> np.ndarray:
    """
    Combine a set of patterns into a single pattern by weighted sum.

    Parameters
    ----------
    patterns : array-like
        2D array of patterns to combine (n_patterns, n_voxels).
    coefficients : array-like, optional
        1D array of coefficients for each pattern (n_patterns). If None, all patterns are weighted equally.

    Returns
    -------
    combined_pattern : array-like
        1D array of the combined pattern (n_voxels).
    """

    if coefficients is None:
        coefficients = np.ones(patterns.shape[1])
    weighted_patterns = patterns * coefficients[:, np.newaxis]
    combined_pattern = np.sum(weighted_patterns, axis=0)
    return combined_pattern
