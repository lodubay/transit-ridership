"""
This script parses and cleans transit data from the various Excel files.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import paths


def main():
    years = list(range(1991, 2020))
    # census_data = import_census_data()
    inflation = import_inflation_data()
    population = uza_population_estimates(years)

    def parse_service_data(sheet, year_range=(1991, 2019), export=True,
                           per_capita=False, inflation_adjusted=False):
        """
        Utility function to parse data from the Service & Operating Expenses
        table.

        Parameters
        ----------
        sheet : str
            Name of sheet in the Excel spreadsheet to import.
        year_range : tuple of ints [default: (1991, 2019)]
            Start and end year (inclusive) of full data.
        export : bool [default: True]
            Whether to export the test and training data to separate CSVs.
        per_capita : bool [default: False]
            If True, normalize time-series data by population each year.
        inflation_adjusted : bool [default: False]
            If True, adjust monetary time-series data for inflation
        
        Returns
        -------
        train : pandas.DataFrame
            Years of training data per city.
        test : pandas.DataFrame
            Year(s) of test data per city.
        
        """
        years = list(range(year_range[0], year_range[1]+1))
        fname = sheet.replace(' ', '_')
        agency_data = read_time_series(
            '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
            sheet,
            codes=list(population.index),
            select_years=years
        )
        city_data = consolidate_city_data(agency_data)
        if inflation_adjusted:
            city_data = adjust_for_inflation(city_data, inflation)
            fname += '_infladj'
        if per_capita:
            city_data = normalize_population(city_data, population)
            fname += '_percap'
        train, test = train_test_split(city_data)
        fname += '.csv'
        if export:
            train.to_csv(paths.data / 'train' / fname)
            test.to_csv(paths.data / 'test' / fname)
        return train, test

    #################################
    ### Vehicle & Ridership Stats ###
    #################################

    # Fare revenue
    parse_service_data('FARES', per_capita=True, inflation_adjusted=True)
    # Vehicles operated in maximum service (VOMS)
    parse_service_data('VOMS', per_capita=True)
    # Vehicle revenue miles (VRM)
    parse_service_data('VRM', per_capita=True)
    # Vehicle revenue hours (VRH)
    parse_service_data('VRH', per_capita=True)
    # Unlinked passenger trips (UPT)
    parse_service_data('UPT', per_capita=True)
    # Passenger Miles Traveled (PMT)
    parse_service_data('PMT', per_capita=True)

    ##########################
    ### Operating Expenses ###
    ##########################

    # Total
    parse_service_data('OpExp Total', per_capita=True, inflation_adjusted=True)

    # Operating expenses
    opexp_total_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp Total',
        codes=list(population.index),
        select_years=years
    )
    year_cols = [c for c in opexp_total_data.columns if is_year(c)]
    other_cols = [c for c in opexp_total_data.columns if not is_year(c)]
    agency_info = opexp_total_data[other_cols].copy()
    agency_info.to_csv(paths.data / 'Agencies.csv')
    opexp_city_data = consolidate_city_data(opexp_total_data)
    # print(adjust_for_inflation(opexp_total_data, inflation))
    opexp_total_data.to_csv(paths.data / 'OpExp_Total.csv')
    # Operating expenses - fraction for vehicle operations (VO)
    # opexp_vo_data = read_time_series(
    #     '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
    #     'OpExp VO',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # opexp_vo_frac = opexp_vo_data.copy()
    # opexp_vo_frac[year_cols] = opexp_vo_data[year_cols] / opexp_total_data[year_cols]
    # opexp_vo_frac.to_csv(paths.data / 'OpExp_VO_Frac.csv')
    # # Operating expenses - fraction for vehicle maintenance (VM)
    # opexp_vm_data = read_time_series(
    #     '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
    #     'OpExp VM',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # opexp_vm_frac = opexp_vm_data.copy()
    # opexp_vm_frac[year_cols] = opexp_vm_data[year_cols] / opexp_total_data[year_cols]
    # opexp_vm_frac.to_csv(paths.data / 'OpExp_VM_Frac.csv')
    # # Operating expenses - fraction for general administration (GA)
    # opexp_ga_data = read_time_series(
    #     '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
    #     'OpExp GA',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # opexp_ga_frac = opexp_ga_data.copy()
    # opexp_ga_frac[year_cols] = opexp_ga_data[year_cols] / opexp_total_data[year_cols]
    # opexp_ga_frac.to_csv(paths.data / 'OpExp_GA_Frac.csv')
    # # Operations funding
    # opfund_data = read_time_series(
    #     '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
    #     'Operating Total',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # opfund_data.to_csv(paths.data / 'OpFund_Total.csv')
    # # Capital funding
    # capfund_data = read_time_series(
    #     '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
    #     'Capital Total',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # capfund_data.to_csv(paths.data / 'CapFund_Total.csv')
    # # Operations funding fraction
    # opfund_frac = opfund_data.copy()
    # opfund_frac[year_cols] = opfund_data[year_cols] / (
    #     opfund_data[year_cols] + capfund_data[year_cols]
    # )
    # opfund_frac.to_csv(paths.data / 'OpFund_Frac.csv')
    # # Federal funding fraction
    # totfund_data = read_time_series(
    #     '2023 TS1.1 Total Funding Time Series_0.xlsx', 
    #     'Total',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # fedfund_data = read_time_series(
    #     '2023 TS1.1 Total Funding Time Series_0.xlsx', 
    #     'Federal',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # fedfund_frac = fedfund_data.copy()
    # fedfund_frac[year_cols] = fedfund_data[year_cols] / totfund_data[year_cols]
    # fedfund_frac.to_csv(paths.data / 'FedFund_Frac.csv')
    # # State funding fraction
    # stfund_data = read_time_series(
    #     '2023 TS1.1 Total Funding Time Series_0.xlsx', 
    #     'Federal',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # stfund_frac = stfund_data.copy()
    # stfund_frac[year_cols] = stfund_data[year_cols] / totfund_data[year_cols]
    # stfund_frac.to_csv(paths.data / 'StateFund_Frac.csv')
    # # Local/other funding fraction
    # locfund_data = read_time_series(
    #     '2023 TS1.1 Total Funding Time Series_0.xlsx', 
    #     'Local',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # otherfund_data = read_time_series(
    #     '2023 TS1.1 Total Funding Time Series_0.xlsx', 
    #     'Other',
    #     codes=list(census_data.index),
    #     select_years=years
    # )
    # locfund_frac = locfund_data.copy()
    # locfund_frac[year_cols] = (
    #     locfund_data[year_cols] + otherfund_data[year_cols]
    # ) / totfund_data[year_cols]
    # locfund_frac.to_csv(paths.data / 'LocalOtherFund_Frac.csv')
    # # Capital expenditures
    # capexp_data = read_time_series(
    #     '2023 TS3.1 Capital Expenditures Time Series.xlsx',
    #     'Total',
    #     codes=list(census_data.index),
    #     select_years=years[1:] # Capital expenditures data begins in 1992
    # )
    # capexp_data.insert(8, '1991', np.nan * np.ones(capexp_data.shape[0]))
    # # Data has multiple entries (different modes) per agency
    # capexp_group = capexp_data.groupby('NTD ID')
    # capexp_sum = capexp_group[year_cols].sum()
    # capexp_data = capexp_data.reset_index().drop_duplicates(subset='NTD ID').drop(columns=['Mode'])
    # capexp_data[year_cols] = capexp_sum[year_cols]
    # capexp_data.set_index('NTD ID', inplace=True)
    # drop_rows = [c for c in capexp_data.index if c not in upt_data.index]
    # capexp_data.drop(index=drop_rows, inplace=True)
    # capexp_data.to_csv(paths.data / 'CapExp_Total.csv')
    # # Estimate population over time
    # cities_in_ts = pd.unique(upt_data['UACE Code']).tolist()
    # pop_est = uza_population_estimates(cities_in_ts, years)
    # pop_est.to_csv(paths.data / 'UZA_population.csv')


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


def adjust_for_inflation(ts_data, inflation_data):
    """
    Adjust time-series monetary data to 2019 dollars.
    
    Parameters
    ----------
    ts_data : pandas.DataFrame
        Un-adjusted time-series data with years as columns.
    inflation_data : pandas.DataFrame
        Value of a 2019 dollar in each year, with years as columns.
    """
    adjusted = ts_data.copy()
    year_cols = [c for c in adjusted.columns if is_year(c)]
    inflation_all = np.tile(inflation_data[year_cols], (adjusted[year_cols].shape[0], 1))
    adjusted[year_cols] *= 1 / inflation_all
    return adjusted


def import_inflation_data():
    """
    Import and properly format inflation data.
    """
    inflation_data = pd.read_csv(paths.data / 'inflation.csv', index_col=0).T
    inflation_data.columns = [str(c) for c in inflation_data.columns]
    return inflation_data


def normalize_population(ts_data, pop_data):
    """
    Divide time-series data by estimated urbanized area population for each year.

    Parameters
    ----------
    ts_data : pandas.DataFrame
        Un-normalized time-series data, with years as columns and a column
        containing UACE city codes.
    pop_data : pandas.DataFrame
        Estimated city populations each year, with years as columns and
        city codes as the index.
    
    Returns
    -------
    pandas.DataFrame
        Time-series data per capita.
    
    """
    year_cols = [c for c in ts_data.columns if is_year(c)]
    data_per_capita = ts_data.copy()
    original_index = data_per_capita.index.name
    if original_index != 'UACE Code':
        data_per_capita.set_index('UACE Code', inplace=True)
    data_per_capita[year_cols] *= 1 / pop_data[year_cols].loc[data_per_capita.index]
    return data_per_capita.set_index(ts_data.index)


def consolidate_city_data(ts_data):
    """
    Take time-series data reported per agency and sum for each urbanized area.

    Parameters
    ----------
    ts_data : pandas.DataFrame
        Time-series data on a per-agency basis. Must have a column titled
        'UACE Data'.

    Returns
    -------
    pandas.DataFrame
        Time-series data summed per city, with 'UACE Code' as the index.
    """
    year_cols = [c for c in ts_data.columns if is_year(c)]
    city_data = ts_data.groupby('UACE Code').sum(year_cols)
    return city_data[year_cols]


def read_time_series(fname, sheet_name, codes=[], dir=paths.data/'time_series',
                     select_years=[], require_all_years=False):
    """
    Import time-series data from Excel file.
    
    Parameters
    ----------
    fname : str
        File name of Excel time-series document.
    sheet_name : str
        Name of Excel sheet to select.
    codes : list of strings [default: []]
        List of UACE codes to limit the data, if provided.
    dir : str or pathlib.Path [default: '../data/time_series/']
        Path to directory containing time-series data.
    select_years : list of ints or strings [default : []]
        Import data from these years only. If empty list, all years with data
        will be included.
    require_all_years : bool [default: False]
        If True, limit to agencies with data available for all years. If 
        select_years is provided, limit to agencies with data all provided
        years.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned and cut time-series data by agency.
    """
    ts_data = pd.read_excel(
        Path(dir) / fname,  sheet_name=sheet_name, 
        dtype={'NTD ID': 'str', 'UACE Code': 'str'}
    )
    # Excel likes to truncate codes with leading 0's... sigh...
    ts_data['UACE Code'] = ts_data['UACE Code'].apply(standardize_uace_code)
    ts_data['NTD ID'] = ts_data['NTD ID'].apply(standardize_uace_code)
    # Limit to actual cities
    ts_data = ts_data[(ts_data['Last Report Year'] == 2023) &
                      (ts_data['Agency Status'] == 'Active') &
                      (ts_data['Reporter Type'] == 'Full Reporter') &
                      (ts_data['Reporting Module'] == 'Urban')].copy()
    # Get data columns
    all_years = [c for c in ts_data.columns if is_year(c)]
    if len(select_years) > 0:
        year_cols = [str(y) for y in select_years if str(y) in ts_data.columns]
        drop_years = [y for y in all_years if not y in year_cols]
        ts_data.drop(columns=drop_years, inplace=True)
    year_cols = [c for c in ts_data.columns if is_year(c)]
    # Limit to cities with data for all years
    if require_all_years:
        ts_data.dropna(axis=0, how='any', subset=year_cols, inplace=True)
    # Remove cities we don't have full census data for
    unique_city_codes = pd.unique(ts_data['UACE Code']).tolist()
    cities_in_census = [c for c in unique_city_codes if c in codes]
    ts_data = ts_data.set_index(
        'UACE Code', drop=False
    ).loc[cities_in_census].set_index('NTD ID')
    ts_data.drop(inplace=True, columns=[
        'Last Report Year', 'Legacy NTD ID', 'Agency Status', 'Reporter Type', 
        'Reporting Module', 'Census Year'
    ])
    try:
        ts_data = ts_data[ts_data['2023 Status'] == 'Existing 2023'].copy()
        ts_data.drop(inplace=True, columns=['2023 Status'])
    except KeyError:
        ts_data = ts_data[ts_data['2023 Mode Status'] == 'Existing 2023'].copy()
        ts_data.drop(inplace=True, columns=['2023 Mode Status'])
    return ts_data


def uza_population_estimates(years, extrapolate=True):
    """
    Produce a time-series of urbanized area population estimates.
    
    Linearly interpolates between available census data and extrapolates
    beyond available data.
    
    Parameters
    ----------
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
    codes = census_data.index.to_list()
    census_years = census_data.columns[1:]
    interp_data = np.zeros((len(codes), len(years)))
    for i, code in enumerate(codes):
        interp_data[i] = np.interp(
            years, census_years, census_data[census_years].loc[code]
        ).astype(int)
    # Package nicely
    df = pd.DataFrame(interp_data, index=pd.Series(codes, name='UACE Code'), 
                      columns=[str(y) for y in years], dtype=int)
    df.insert(0, 'Name', census_data['Name'].loc[codes])
    return df


def import_census_data(dir=paths.data/'census'):
    """
    Import all years of census data as DataFrames.
    
    Parameters
    ----------
    dir : str or pathlib.Path [default: ``../data/census/``]
        Directory path containing census data files.
    
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
    """
    Add leading '0's to the given code to make it 5 digits.
    """
    code = str(code)
    code = '0' * (5 - len(code)) + code
    return code


def is_year(val):
    """
    Determine if the given string represents a year.
    """
    match = re.match(r'([1-2][0-9]{3})', val)
    return (match is not None) and (len(val) == 4)


if __name__ == '__main__':
    main()
