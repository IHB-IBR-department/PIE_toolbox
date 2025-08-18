import pandas as pd
import numpy as np
from pie_toolbox.core.validators import check_data


def join_headers(rows_header, sep='_'):
    """
    Join column header names from a pandas DataFrame to a single string.

    Parameters
    ----------
    rows_header : pandas.Index
        Index of the header of the DataFrame.
    sep : str, optional
        Separator to join names. Default is '_'.

    Returns
    -------
    str
        Joined column name, like 'Header(Level3)_Header(Level2)_Header(Level1)'

    Notes
    -----
    If there is only one header, this header is returned.
    If the header is unnamed, this header is skipped.
    If there are no header names, returns 'Unnamed'.
    """
    res = None
    for i_row in range(len(rows_header) - 1, -1, -1):
        if not ('Unnamed:' in str(rows_header[i_row])):
            if (res is None):
                res = str(rows_header[i_row])
            else:
                res += sep + str(rows_header[i_row])
    if (res is None):
        res = 'Unnamed'
    return res


class Table():
    filename_column = ""

    def __init__(self, dataframe: pd.DataFrame = None,
                 filename_column: str = ""):
        self.dataframe = dataframe
        self.filename_column = filename_column

    def open_excel(self, filepath: str, header_num: int = 1, sep: str = '_'):
        """
        Open an Excel table for classification.

        Parameters
        ----------
        filepath : str
            Path to the Excel file.
        header_num : int, optional
            Number of header rows. Default is 1.
        sep : str, optional
            Separator to join header names. Default is '_'.

        """

        check_data.check_type(filepath, str)
        check_data.check_type(header_num, int)
        if (header_num < 1):
            raise ValueError("Header number must be greater than 0")
        check_data.check_not_none_or_empty(filepath)
        tb = pd.read_excel(filepath, header=list(np.arange(header_num)))
        tb.columns = [join_headers(rows_header, sep=sep)
                      for rows_header in tb.columns]
        self.dataframe = tb

    def get_labels(self, column_name: str, filenames: list = []) -> np.ndarray:
        """
        Get labels from the table.

        Parameters
        ----------
        column_name : str
            Name of the column with labels.
        filenames : list of str, optional
            Filenames to filter labels. Default is an empty list.

        Returns
        -------
        labels : numpy.ndarray
            Array of labels as strings.
        """

        check_data.check_not_none_or_empty(self.dataframe)

        column_data = self.dataframe[column_name]

        if (len(filenames) > 0) and (len(self.filename_column) > 0):
            column_data = column_data[self.dataframe[self.filename_column].isin(
                filenames)]

        return column_data.astype(str).to_numpy()


def set_values(tb, set=[], value=0):
    tb_mod = tb.fillna(0).copy()
    for iter in set:
        tb_mod[tb_mod == iter] = value
    return tb_mod
