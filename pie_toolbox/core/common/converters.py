import numpy as np
import os
from pie_toolbox.core.validators import check_data


def convert_index(indexes: str | int | list):
    '''
    Input:
        indexes: index (str or int) or list of indexes (str or array)
    Output:
        array of indexes (ndarray)
    '''
    if indexes is None:
        return np.array([])
    if isinstance(indexes, str):
        return np.fromstring(indexes, dtype=int, sep=' ')
    elif isinstance(indexes, int):
        return np.array([indexes])
    else:
        return np.array(indexes)


def files_list(files: str | list, file_type: str = "nii"):
    """
    Input:
        files: single file, list of files, single folder or list of folders
        file_type: type of files

    Output:
        list of files
    """

    if file_type[0] != ".":
        file_type = "." + file_type

    def open_folder(folder_name):
        return [os.path.join(folder_name, element) for element in os.listdir(
            folder_name) if file_type in element]

    if isinstance(files, str):
        if (file_type in files):
            return [files]
        elif not "." in files[-4:]:
            return (open_folder(files))
        else:
            return None

    files_result = [element for element in files if file_type in element]
    folders = [element for element in files if not (
        "." in element.split("\\")[-1])]
    for folder in folders:
        try:
            files_result.extend(open_folder(folder))
        except BaseException:
            pass

    return files_result


def get_scores_from_list(scores: list, region_index: int = 0) -> np.ndarray:
    """
    Convert a list of scores to a single numpy array with shape (n_subjects, n_scores)

    Parameters
    ----------
    scores : list
        A list of scores with length n_patterns, each score is a numpy array with shape (n_subjects, n_regions)
    region_index : int, optional
        The index of the region to extract from each score. If not provided, defaults to 0.

    Returns
    -------
    scores_array : np.ndarray
        A numpy array with shape (n_subjects, n_patterns) containing the selected region of each score

    Raises
    ------
    ValueError
        If scores is not a list of numpy arrays with shape (n_subjects, n_regions)
    """
    if isinstance(scores, list) and all(isinstance(score, np.ndarray)
                                        for score in scores):
        n_subjects = scores[0].shape[0]
        if not all(score.shape[0] == n_subjects for score in scores):
            raise ValueError(
                "Each score must have the same number of subjects")
        if not all(region_index < score.shape[1] for score in scores):
            raise ValueError("The region_index must be valid for all scores")
        return np.array([score[:, region_index] for score in scores]).T
    raise ValueError(
        "Scores must be a list of numpy arrays with shape (n_subjects, n_regions)")


def get_concatenated_scores(scores: list) -> np.ndarray:
    """
    Concatenate a list of scores arrays into a single array with shape (n_subjects, n_scores)

    Parameters
    ----------
    scores : list
        A list of score arrays, each array with shape (n_subjects, n_regions)

    Returns
    -------
    scores_concatenated : np.ndarray
        A single numpy array with shape (n_subjects, n_scores) containing the concatenated scores

    Raises
    ------
    ValueError
        If scores is not a list of numpy arrays with shape (n_subjects, n_regions)
    """
    check_data.check_type(scores, list)
    check_data.check_type(scores[0], np.ndarray)
    n_subjects = scores[0].shape[0]
    if not all(score.shape[0] == n_subjects for score in scores):
        raise ValueError(
            "Each score array must have the same number of subjects")
    scores_concatenated = np.concatenate(
        [scores_pattern for scores_pattern in scores], axis=1)
    return scores_concatenated


def labels_to_indexes(labels: np.ndarray) -> np.ndarray:
    """
    Convert a numpy array of labels to a numpy array of indexes.

    Parameters
    ----------
    labels : np.ndarray
        A numpy array of labels.

    Returns
    -------
    indexes : np.ndarray
        A numpy array of indexes, where each index is the index of the label in the sorted list of unique labels.
    """

    return np.unique(labels, return_inverse=True)[1]


def get_whole_brain_scores(scores: list | np.ndarray):
    """
    Get the scores of the given pattern in the given data, assuming whole brain computation.

    Parameters
    ----------
    scores : list|np.ndarray
        A list of score arrays, each array with shape (n_subjects, n_regions),
        or a numpy array with shape (n_subjects, n_regions)

    Returns
    -------
    scores_array : np.ndarray
        A numpy array with shape (n_subjects,) containing the scores of the given pattern in the given data

    Raises
    ------
    ValueError
        If scores is not a list of numpy arrays with shape (n_subjects, 1)
        or if scores is not a numpy array with shape (n_subjects, 1)
    """

    if isinstance(scores, list):
        # if all scores inside list have shape (n_subjects, 1), i.e. containing
        # only one region
        if not all(isinstance(score, np.ndarray) for score in scores):
            raise ValueError("All score arrays must be numpy arrays")
        if all(score.shape[1] == 1 for score in scores):
            scores = get_scores_from_list(scores, region_index=0)
        else:
            raise ValueError(
                "All score arrays must have shape (n_subjects, 1), i.e. containing only one region")
    elif isinstance(scores, np.ndarray):
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[0, :]
        elif scores.ndim > 2:
            raise ValueError("Scores must be 1D array")
    else:
        raise TypeError("Scores must be a list or numpy array")
    return scores


def add_extension(filename: str, extension: str) -> str:
    """
    Add the given extension to the given filename if not already present

    Parameters
    ----------
    filename : str
        The filename to add the extension to
    extension : str
        The extension to add

    Returns
    -------
    str
        The filename with the extension added if not already present
    """
    if not filename.endswith('.' + extension):
        return filename + '.' + extension
    return filename
