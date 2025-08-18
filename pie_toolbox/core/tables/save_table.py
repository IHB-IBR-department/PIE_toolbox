from pie_toolbox.core.classifier import metrics as ssm_pca_metrics
from pie_toolbox.core.classifier.metrics import Metrics
from pie_toolbox.core.validators import check_data
from pie_toolbox.core.common import converters
from pie_toolbox.core.classifier .cross_val import CrossValResults
import pandas as pd
import numpy as np


def get_dataframe(metrics: Metrics, information: str = ""):
    """
    Creates a pandas DataFrame from the given Metrics object.

    Parameters
    ----------
    metrics : Metrics
        The Metrics object from which to create the DataFrame.
    information : str, optional
        Any additional information to include in the DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the confusion matrix, sensitivity, specificity, accuracy, balanced accuracy, PPV, NPV, and AUC.
    """
    if isinstance(metrics, CrossValResults):
        metrics = metrics.get_metrics()

    labels_uniq = np.unique(metrics.real_labels)
    df = pd.DataFrame(metrics.confusion_matrix, columns=labels_uniq,
                      index=pd.MultiIndex.from_product([['Actual'], labels_uniq]), dtype='object')
    df.loc['Sensitivity', labels_uniq] = metrics.sensitivity
    df.loc['Information', labels_uniq[0]] = str(information)
    df.loc['Specificity', labels_uniq] = metrics.specificity
    df.loc['Accuracy', labels_uniq] = metrics.accuracy
    df.loc['Balanced accuracy', labels_uniq] = metrics.balanced_accuracy
    df.loc['PPV', labels_uniq] = metrics.ppv
    df.loc['NPV', labels_uniq] = metrics.npv
    df.loc['AUC', labels_uniq] = metrics.auc
    return df


def save_excel(dataframe: list | pd.DataFrame,
               filepath: str, sheet_name="Model"):
    """
    Save a DataFrame or list of DataFrames to an Excel file.

    Parameters
    ----------
    dataframe : list or pd.DataFrame
        The DataFrame or list of DataFrames to save to the Excel file.
    filepath : str
        The path where the Excel file will be saved.
    sheet_name : str, optional
        Base name for the Excel sheet. Default is "Model". Each DataFrame in the list will be saved on a separate sheet.

    Notes
    -----
    If a single DataFrame is provided, it will be wrapped in a list and saved to one sheet.
    If a list of DataFrames is provided, each DataFrame will be saved to a separate sheet.
    """

    if isinstance(dataframe, pd.DataFrame):
        dataframe = [dataframe]
    check_data.check_type(dataframe, list)
    filepath = converters.add_extension(filepath, 'xlsx')
    with pd.ExcelWriter(filepath) as ex_wr:
        for i, cm in enumerate(dataframe):
            cm.to_excel(ex_wr, sheet_name=sheet_name + ' ' + str(i + 1))


def save_csv(dataframe: list | pd.DataFrame, filepath: str):
    """
    Save a DataFrame or list of DataFrames to a CSV file.

    Parameters
    ----------
    dataframe : list or pd.DataFrame
        The DataFrame or list of DataFrames to save to the CSV file.
    filepath : str
        The path where the CSV file will be saved.

    Notes
    -----
    Each DataFrame in the list will be saved to a separate CSV file.
    """
    filepath = converters.add_extension(filepath, 'csv')
    if isinstance(dataframe, list):
        for i, df in enumerate(dataframe):
            df.to_csv(filepath[:-4] + f"_{i+1}" + filepath[-4:], index=True)
    else:
        check_data(dataframe, pd.DataFrame)
        dataframe.to_csv(filepath, index=True)
