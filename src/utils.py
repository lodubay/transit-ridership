"""
This file contains general utility functions for the repository.
"""

import re
import numpy as np
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


def read_data(feature, dir='train'):
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
        paths.data / dir / ('%s.csv' % feature), 
        index_col='UACE Code', 
        dtype={'UACE Code': str}
    )


def is_year(val):
    """
    Determine if the given string represents a year.
    """
    match = re.match(r'([1-2][0-9]{3})', val)
    return (match is not None) and (len(val) == 4)


def train_test_year_split(data, n_years=1):
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


def train_test_city_split(data, test_size=0.2, seed=2024):
    """
    Randomly split data into training and testing cities.
    
    Note: train/test split is done by city and not by year.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data with all features and indexed by city.
    test_size : float, optional [default: 0.2]
        Fraction of cities to use for testing.
    
    Returns
    -------
    train : pandas.DataFrame
    test : pandas.DataFrame
    
    """
    rng = np.random.default_rng(seed=seed)
    all_cities = np.unique(data.index.to_numpy())
    test_cities = rng.choice(
        all_cities, size=int(test_size * all_cities.shape[0]), replace=False
    )
    train_cities = np.array([c for c in all_cities if not c in test_cities])
    assert test_cities.shape[0] + train_cities.shape[0] == all_cities.shape[0]
    return data.loc[train_cities,:], data.loc[test_cities,:]


def consolidate_features(features, dir='train'):
    """
    Combine multiple features into a single dataframe.
    
    Note: Resulting DataFrame is indexed by city, with year as an additional 
    feature.

    Parameters
    ----------
    features : list of strings
        List of feature names to import and combine.
    
    Returns
    -------
    pandas.DataFrame
        Dataframe indexed by city, with year as an additional feature.
    """
    df = pd.concat(
        [read_data(feature, dir=dir).stack() for feature in features],
        axis=1
    )
    df.columns = features
    df.index.names = ['UACE Code', 'Year']
    # Turn year into a feature
    df = df.reset_index().set_index('UACE Code')
    df = df.astype({'Year': int})
    return df
