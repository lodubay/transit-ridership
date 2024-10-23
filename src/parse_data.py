"""
This script parses and cleans transit data from the various Excel files.
"""

import numpy as np
import pandas as pd
import paths


def main():
    years = list(range(1991, 2024))
    upt_data = pd.read_excel(paths.data / 'time_series' / '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
                             sheet_name='UPT', dtype={'NTD ID': 'str', 'UACE Code': 'str'})
    # Excel likes to truncate codes with leading 0's... sigh...
    upt_data['UACE Code'] = upt_data['UACE Code'].apply(standardize_uace_code)
    upt_data['NTD ID'] = upt_data['NTD ID'].apply(standardize_uace_code)
    # Limit to actual cities
    upt_data = upt_data[(upt_data['Last Report Year'] == 2023) &
                        (upt_data['Agency Status'] == 'Active') &
                        (upt_data['Reporter Type'] == 'Full Reporter') &
                        (upt_data['Reporting Module'] == 'Urban')].copy()
    # Remove cities we don't have full census data for
    unique_city_codes = pd.unique(upt_data['UACE Code']).tolist()
    census_data = import_census_data()
    cities_in_census = [c for c in unique_city_codes if c in census_data.index]
    upt_data = upt_data.set_index('UACE Code').loc[cities_in_census].set_index('NTD ID')
    upt_data.drop(inplace=True, columns=[
        'Last Report Year', 'Legacy NTD ID', 'Agency Status', 'Reporter Type', 
        'Reporting Module', 'Census Year', '2023 Status'
    ])
    upt_data.to_csv(paths.data / 'UPT.csv')
    # Estimate population over time
    pop_est = uza_population_estimates(cities_in_census, years)
    pop_est.to_csv(paths.data / 'UZA_population.csv')


def uza_population_estimates(codes, years, extrapolate=True):
    """
    Produce a time-series of urbanized area population estimates.
    
    Linearly interpolates between available census data and extrapolates
    beyond available data.
    
    Parameters
    ----------
    codes : list of strings
        List of UACE codes (i.e., city identifiers).
    years : list of ints
        List of years to estimate population.
    extrapolate : bool [default: True]
        Extrapolate beyond the end of available census data in both directions.
    
    Returns
    -------
    pandas.DataFrame
        Population estimates, with UACE as the index and years as column names.
    
    """
    census_data = import_census_data()
    # Estimate population beyond available census data
    if extrapolate:
        pred1990 = census_data[2000] - (census_data[2010] - census_data[2000])
        census_data.insert(1, 1990, pred1990)
        pred2030 = census_data[2020] + (census_data[2020] - census_data[2010])
        census_data[2030] = pred2030
    census_data = census_data.loc[codes].copy()
    census_years = census_data.columns[1:]
    interp_data = np.zeros((len(codes), len(years)))
    for i, code in enumerate(codes):
        interp_data[i] = np.interp(
            years, census_years, census_data[census_years].loc[code]
        ).astype(int)
    # Package nicely
    df = pd.DataFrame(interp_data, index=pd.Series(codes, name='UACE'), columns=years, dtype=int)
    df.insert(0, 'Name', census_data['Name'].loc[codes])
    return df


def import_census_data(dir=paths.data/'census', extrapolate_left = False,
                       extrapolate_right = False):
    """
    Import all years of census data as DataFrames.
    
    Parameters
    ----------
    dir : str or pathlib.Path [default: ``../data/census/``]
        Directory path containing census data files.
    exterapolate_left : bool [default: False]
        If True, generate a DataFrame 10 years before the first available
        assuming population growth extrapolated from the first 10 years.
    extrapolate_right : bool [default: False]
        If True, generate a DataFrame 10 years after the most recent available
        with population growth extrapolated from the previous 10 years.
    
    Returns
    -------
    pandas.DataFrame
        Population of each urban area for all census years.
    
    """
    # Import available census files
    remove_comma = lambda s: s.replace(',', '')
    census2000 = pd.read_csv(
        dir / '2000_ua_list.csv', 
        header=0, index_col=0, usecols=[0, 1, 2],
        names=['Code', 'Name', 'Population'],
        dtype={'Code': 'str'}
    )
    census2010 = pd.read_csv(
        dir / '2010_ua_list_all.csv',
        header=0, index_col=0, usecols=[0, 1, 2],
        names=['Code', 'Name', 'Population'],
        converters={'Population': remove_comma},
        dtype={'Code': 'str'}
    )
    census2010['Population'] = census2010['Population'].astype('int')
    census2020 = pd.read_csv(
        dir / '2020_ua_list_all.csv',
        header=0, index_col=0, usecols=[0, 1, 2],
        names=['Code', 'Name', 'Population'],
        converters={'Population': remove_comma},
        dtype={'Code': 'str'}
    )
    census2020['Population'] = census2020['Population'].astype('int')
    # Unify city code lists (remove cities not in all census years)
    populations = pd.concat(
        [census2000[['Population']], census2010[['Population']], 
         census2020[['Population']]], join='inner', axis=1
    )
    populations.columns = [2000, 2010, 2020]
    # Insert column with city name
    populations.insert(0, 'Name', census2020['Name'])
    return populations


def standardize_uace_code(code):
    code = str(code)
    code = '0' * (5 - len(code)) + code
    return code


if __name__ == '__main__':
    main()
