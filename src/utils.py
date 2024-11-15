"""
This file contains general utility functions for the repository.
"""

import re
import pandas as pd
import paths


def get_city_code(name, city_info=None):
    """
    Convert urbanized area name to UACE code.
    
    Parameters
    ----------
    name : str
        Name of urbanized area.
    city_info : pandas.DataFrame, optional
        DataFrame containing city names and codes. If None, import from 
        ../data/Cities.csv
    
    Returns
    -------
    str
        UACE code.
    """
    if city_info is None:
        city_info = pd.read_csv(
            paths.data / 'Cities.csv', 
            index_col='UACE Code', 
            dtype={'UACE Code': str}
        )
    return city_info[city_info['Primary UZA Name'] == name].index[0]

def get_city_name(code, city_info=None):
    """
    Convert UACE code to urbanized area name.
    
    Parameters
    ----------
    code : str
        Five-digit UACE code, including leading zeroes.
    city_info : pandas.DataFrame, optional
        DataFrame containing city names and codes. If None, import from 
        ../data/Cities.csv
    
    Returns
    -------
    str
        Urbanized area name.
    """
    if city_info is None:
        city_info = pd.read_csv(
            paths.data / 'Cities.csv', 
            index_col='UACE Code', 
            dtype={'UACE Code': str}
        )
    return city_info.loc[code]['Primary UZA Name'].iloc[0]


def read_training_data(feature):
    """
    Import parsed training data for all cities.

    Parameters
    ----------
    feature : str
        Name of the parsed training data file, excluding the extension.
    
    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(
        paths.data / 'train' / ('%s.csv' % feature), 
        index_col='UACE Code', 
        dtype={'UACE Code': str}
    )


def read_testing_data(feature):
    """
    Import parsed training data for all cities.

    Parameters
    ----------
    feature : str
        Name of the parsed training data file, excluding the extension.
    
    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(
        paths.data / 'test' / ('%s.csv' % feature), 
        index_col='UACE Code', 
        dtype={'UACE Code': str}
    )


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
