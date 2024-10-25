"""
This script parses and cleans transit data from the various Excel files.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import paths


def main():
    census_data = import_census_data()
    years = list(range(1991, 2020))
    # Operating expenses
    opexp_total_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp Total',
        codes=list(census_data.index),
        select_years=years
    )
    opexp_total_data.to_csv(paths.data / 'OpExp_Total.csv')
    year_cols = [c for c in opexp_total_data.columns if is_year(c)]
    # Operating expenses - fraction for vehicle operations (VO)
    opexp_vo_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp VO',
        codes=list(census_data.index),
        select_years=years
    )
    opexp_vo_frac = opexp_vo_data.copy()
    opexp_vo_frac[year_cols] = opexp_vo_data[year_cols] / opexp_total_data[year_cols]
    opexp_vo_frac.to_csv(paths.data / 'OpExp_VO_Frac.csv')
    # Operating expenses - fraction for vehicle maintenance (VM)
    opexp_vm_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp VM',
        codes=list(census_data.index),
        select_years=years
    )
    opexp_vm_frac = opexp_vm_data.copy()
    opexp_vm_frac[year_cols] = opexp_vm_data[year_cols] / opexp_total_data[year_cols]
    opexp_vm_frac.to_csv(paths.data / 'OpExp_VM_Frac.csv')
    # Operating expenses - fraction for general administration (GA)
    opexp_ga_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp GA',
        codes=list(census_data.index),
        select_years=years
    )
    opexp_ga_frac = opexp_ga_data.copy()
    opexp_ga_frac[year_cols] = opexp_ga_data[year_cols] / opexp_total_data[year_cols]
    opexp_ga_frac.to_csv(paths.data / 'OpExp_GA_Frac.csv')
    # Fare revenue
    fare_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'FARES',
        codes=list(census_data.index),
        select_years=years
    )
    fare_data.to_csv(paths.data / 'FARES.csv')
    # Vehicles operated in maximum service (VOMS)
    voms_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VOMS',
        codes=list(census_data.index),
        select_years=years
    )
    voms_data.to_csv(paths.data / 'VOMS.csv')
    # Vehicle revenue miles (VRM)
    vrm_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VRM',
        codes=list(census_data.index),
        select_years=years
    )
    vrm_data.to_csv(paths.data / 'VRM.csv')
    # Vehicle revenue hours (VRH)
    vrh_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VRH',
        codes=list(census_data.index),
        select_years=years
    )
    vrh_data.to_csv(paths.data / 'VRH.csv')
    # Unlinked passenger trips (UPT)
    upt_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'UPT',
        codes=list(census_data.index),
        select_years=years
    )
    upt_data.to_csv(paths.data / 'UPT.csv')
    # Unlinked passenger trips (UPT)
    pmt_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'PMT',
        codes=list(census_data.index),
        select_years=years
    )
    pmt_data.to_csv(paths.data / 'PMT.csv')
    # Operations funding
    opfund_data = read_time_series(
        '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
        'Operating Total',
        codes=list(census_data.index),
        select_years=years
    )
    opfund_data.to_csv(paths.data / 'OpFund_Total.csv')
    # Capital funding
    capfund_data = read_time_series(
        '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
        'Capital Total',
        codes=list(census_data.index),
        select_years=years
    )
    capfund_data.to_csv(paths.data / 'CapFund_Total.csv')
    # Operations funding fraction
    opfund_frac = opfund_data.copy()
    opfund_frac[year_cols] = opfund_data[year_cols] / (
        opfund_data[year_cols] + capfund_data[year_cols]
    )
    opfund_frac.to_csv(paths.data / 'OpFund_Frac.csv')
    # Federal funding fraction
    totfund_data = read_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx', 
        'Total',
        codes=list(census_data.index),
        select_years=years
    )
    fedfund_data = read_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx', 
        'Federal',
        codes=list(census_data.index),
        select_years=years
    )
    fedfund_frac = fedfund_data.copy()
    fedfund_frac[year_cols] = fedfund_data[year_cols] / totfund_data[year_cols]
    fedfund_frac.to_csv(paths.data / 'FedFund_Frac.csv')
    # State funding fraction
    stfund_data = read_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx', 
        'Federal',
        codes=list(census_data.index),
        select_years=years
    )
    stfund_frac = stfund_data.copy()
    stfund_frac[year_cols] = stfund_data[year_cols] / totfund_data[year_cols]
    stfund_frac.to_csv(paths.data / 'StateFund_Frac.csv')
    # Local/other funding fraction
    locfund_data = read_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx', 
        'Local',
        codes=list(census_data.index),
        select_years=years
    )
    otherfund_data = read_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx', 
        'Other',
        codes=list(census_data.index),
        select_years=years
    )
    locfund_frac = locfund_data.copy()
    locfund_frac[year_cols] = (
        locfund_data[year_cols] + otherfund_data[year_cols]
    ) / totfund_data[year_cols]
    locfund_frac.to_csv(paths.data / 'LocalOtherFund_Frac.csv')
    # Capital expenditures
    capexp_data = read_time_series(
        '2023 TS3.1 Capital Expenditures Time Series.xlsx',
        'Total',
        codes=list(census_data.index),
        select_years=years[1:] # Capital expenditures data begins in 1992
    )
    capexp_data.insert(8, '1991', np.nan * np.ones(capexp_data.shape[0]))
    # Data has multiple entries (different modes) per agency
    capexp_group = capexp_data.groupby('NTD ID')
    capexp_sum = capexp_group[year_cols].sum()
    capexp_data = capexp_data.reset_index().drop_duplicates(subset='NTD ID').drop(columns=['Mode'])
    capexp_data[year_cols] = capexp_sum[year_cols]
    capexp_data.set_index('NTD ID', inplace=True)
    drop_rows = [c for c in capexp_data.index if c not in upt_data.index]
    capexp_data.drop(index=drop_rows, inplace=True)
    capexp_data.to_csv(paths.data / 'CapExp_Total.csv')
    # Estimate population over time
    cities_in_ts = pd.unique(upt_data['UACE Code']).tolist()
    pop_est = uza_population_estimates(cities_in_ts, years)
    pop_est.to_csv(paths.data / 'UZA_population.csv')


def read_time_series(fname, sheet_name, codes=[], dir=paths.data/'time_series',
                     select_years=[], require_all_years=True):
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
    require_all_years : bool [default: True]
        If True, limit to agencies with data available for all years. If 
        select_years is provided, limit to agencies with data all provided
        years.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned and cut time-series data.
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
    df = pd.DataFrame(interp_data, index=pd.Series(codes, name='UACE Code'), 
                      columns=years, dtype=int)
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
    code = str(code)
    code = '0' * (5 - len(code)) + code
    return code


def is_year(val):
    match = re.match(r'([1-2][0-9]{3})', val)
    return (match is not None) and (len(val) == 4)


if __name__ == '__main__':
    main()
