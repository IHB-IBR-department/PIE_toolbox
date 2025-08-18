from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import numpy as np
from sklearn.metrics import roc_curve, auc
from pie_toolbox.core.common.logger import setup_root_logger, get_logger

setup_root_logger('cross_val')
logger = get_logger('cross_val')


def get_random_state():

    return np.random.randint(1, 999)


def get_cv_split(data: np.ndarray, labels: np.ndarray,
                 split_type, folds=5, random_state=None):
    """
    Generates cross-validation splits for the given data and labels.

    Parameters:
    data (np.ndarray): The data to be split for cross-validation.
    labels (np.ndarray): The labels corresponding to the data.
    split_type (str): The type of cross-validation to use. Options are "LOO" for Leave-One-Out
                      or "Folds" for StratifiedKFold.
    folds (int, optional): Number of folds for StratifiedKFold. Default is 5.
    random_state (int, optional): The seed used by the random number generator. Default is None,
                                  which uses a random seed.

    Returns:
    generator: A generator yielding train/test splits.
    int: The number of splits.
    """

    if random_state is None:
        rand_state = get_random_state()
    else:
        rand_state = random_state

    split_mechanism = {"LOO": LeaveOneOut(),
                       "Folds": StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
                       }[split_type]
    return split_mechanism.split(
        data, labels), split_mechanism.get_n_splits(data, labels)


def get_auc(real_labels, predicted_scores, class_name=None):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Parameters
    ----------
    real_labels : array-like of shape (n_samples,)
        Real labels of the samples.
    predicted_scores : array-like of shape (n_samples,)
        Predicted scores or probabilities for the positive class.
    class_name : optional, default=None
        Class label for the positive class.

    Returns
    -------
    roc_auc : float
        Area under the ROC curve.
    fpr : array-like
        False positive rates.
    tpr : array-like
        True positive rates.
    """

    labels_uniq = np.unique(real_labels)
    if class_name is None:
        pos_label = labels_uniq[1]
    else:
        pos_label = class_name
    fpr, tpr, _ = roc_curve(
        real_labels, predicted_scores[:, labels_uniq == pos_label], pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def fix_bad_groups(
        predict_proba_score: np.ndarray,
        test_labels: np.ndarray,
        train_labels: np.ndarray,
        all_labels: np.ndarray) -> np.ndarray:
    """
    Adjusts the predicted probability scores by inserting zero probabilities for
    test labels that are not present in the train labels.

    Parameters
    ----------
    predict_proba_score : np.ndarray
        The predicted probability scores for the test set.
    test_labels : np.ndarray
        The labels of the test set.
    train_labels : np.ndarray
        The labels of the training set.
    all_labels : np.ndarray
        All possible labels in the dataset.

    Returns
    -------
    np.ndarray
        The adjusted predicted probability scores with zero columns added for
        missing test labels.

    Example
    -------
    Suppose:
    predict_proba_score = [0.7, 0.1, 0.2]
    test_labels = ['dog']
    train_labels = ['cat', 'fish', 'bird']
    all_labels = ['cat', 'dog', 'fish', 'bird']

    Since 'dog' is not in train_labels, the returned array will include
    a zero probability for 'dog':
    [0.7, 0.0, 0.1, 0.2]

    Notes
    -------
    If this function outputs a warning, it means that there is a label in the test set
    that is not present in the training set. You should check your data for inconsistencies (groups with single sample).
    """

    if not np.isin(test_labels, train_labels).all():
        logger.warning(
            f"Test labels {test_labels} not present in train labels {train_labels}. It means that the group is too small. "
            "Inserting zero probabilities for missing labels."
        )
        for i_label in range(len(test_labels)):
            if test_labels[i_label] not in train_labels:
                insert_pos = np.where(
                    np.unique(all_labels) == test_labels[i_label])[0][0]
                predict_proba_score = np.insert(
                    predict_proba_score, insert_pos, 0, axis=1)
    return predict_proba_score
