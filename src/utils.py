"""
This file contains general utility functions for the repository.
"""

import re
import pandas as pd


def is_year(val):
    """
    Determine if the given string represents a year.
    """
    match = re.match(r'([1-2][0-9]{3})', val)
    return (match is not None) and (len(val) == 4)


def train_test_split(data, n_years=1):
    """
    Split the last N years off as a testing set.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Time-series data with years as columns. Other columns are discarded.
    n_years : int [default: 1]
        Number of years at the end to reserve for testing.
    
    Returns
    -------
    train : pandas.DataFrame
        Time-series data for training
    test : pandas.DataFrame
        Time-series data for testing.
    
    """
    year_cols = [c for c in data.columns if is_year(c)]
    return data[year_cols[:-n_years]].copy(), data[year_cols[-n_years:]].copy()
