"""
This file contains a model which is trained on data from most cities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from utils import read_training_data
import paths

def main():
    all_features = [
        'VRM_percap', 
        'VRH_percap', 
        'VOMS_percap', 
        'OpExp_GA_frac', 
        'OpExp_VM_frac', 
        'OpExp_VO_frac',
        'OpExp_Total_infladj_percap', 
        'OpFund_infladj_percap', 
        'CapFund_infladj_percap', 
        'OpFund_frac', 
        'TotalFund_infladj_percap', 
        'FedFund_frac', 
        'StateFund_frac', 
        'LocalFund_frac', 
    ]
    ycol = 'UPT_percap'
    all_data = consolidate_features(all_features + [ycol])
    all_features = ['Year'] + all_features
    train, test = train_test_split(all_data)
    # EDA - all feature pairs
    sns.pairplot(train, kind='hist', corner=True)
    plt.savefig(paths.plots / 'all_feature_pairs.png', dpi=300)
    # EDA - just y vs all features
    sns.set_style('whitegrid')
    g = sns.FacetGrid(pd.DataFrame(all_features), col=0, col_wrap=4, sharex=False)
    for ax, x_var in zip(g.axes, all_features):
        sns.histplot(data=all_data, x=x_var, y=ycol, ax=ax)
    g.tight_layout()
    plt.savefig(paths.plots / 'all_features.png', dpi=300)
    # Select features
    select_features = [
        'Year',
        'VRH_percap',
        'TotalFund_infladj_percap',
        'FedFund_frac',
        'StateFund_frac',
        'LocalFund_frac',
        'OpExp_GA_frac',
        'OpExp_VM_frac',
        'OpExp_VO_frac',
    ]
    g = sns.FacetGrid(pd.DataFrame(select_features), col=0, col_wrap=5, sharex=False)
    for ax, x_var in zip(g.axes, select_features):
        sns.histplot(data=all_data, x=x_var, y=ycol, ax=ax)
    g.tight_layout()
    plt.savefig(paths.plots / 'select_features.png', dpi=300)
    plt.close()
    # Baseline model: mean ridership
    baseline_pred = train['UPT_percap'].mean() * np.ones(test.shape[0])
    baseline_rmse = root_mean_squared_error(test['UPT_percap'], baseline_pred)
    # Regression pipeline
    lasso_pipe = Pipeline([
        ('scale', StandardScaler()),  
        ('lasso', LassoCV(alphas=None, cv=5, max_iter=100000))  
    ])
    X = train[select_features].values
    y = train['UPT_percap'].values.ravel()
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)  # filter out nans
    X = X[mask]
    y = y[mask]
    lasso_pipe.fit(X, y)
    print(lasso_pipe['lasso'].coef_)
    print(lasso_pipe['lasso'].intercept_)
    # Test on 1991-2018 data
    lasso_pred = lasso_pipe.predict(test[select_features])
    lasso_rmse = root_mean_squared_error(test['UPT_percap'], lasso_pred)
    print(baseline_rmse)
    print(lasso_rmse)
    # Plot results
    fig, axs = plt.subplots(2, sharex=True, figsize=(4, 8), tight_layout=True)
    axs[0].plot(lasso_pred, test['UPT_percap'], '.')
    bounds = [lasso_pred.min(), lasso_pred.max()]
    axs[0].plot(bounds, bounds, 'k--')
    axs[0].set_ylabel('Test data')
    axs[0].set_title('Unlinked Passenger Trips per Capita')
    axs[1].plot(lasso_pred, test['UPT_percap'] - lasso_pred, '.')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Residual')
    plt.savefig(paths.plots / 'test_mlr_predict.png', dpi=300)
    plt.close()
    # Naive baseline predictions
    # naive_forecast = train[train['Year'] == 2018]['UPT_percap']


def consolidate_features(features):
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
        [read_training_data(feature).stack() for feature in features],
        axis=1
    )
    df.columns = features
    df.index.names = ['UACE Code', 'Year']
    # Turn year into a feature
    df = df.reset_index().set_index('UACE Code')
    df = df.astype({'Year': int})
    return df

def train_test_split(data, test_size=0.2, seed=2024):
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


if __name__ == '__main__':
    main()
