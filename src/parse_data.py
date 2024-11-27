"""
This script parses and cleans transit data from the various Excel files.
"""

from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import paths
from utils import is_year, train_test_year_split


def main():
    years = list(range(1991, 2020))
    inflation = import_inflation_data()
    population = uza_population_estimates(years)

    # Agency and city reference info
    upt_data = read_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'UPT',
        codes=list(population.index),
        select_years=years
    )
    year_cols = [c for c in upt_data.columns if is_year(c)]
    other_cols = [c for c in upt_data.columns if not is_year(c)]
    # Limit to cities with ridership data for all years
    upt_city_data = consolidate_city_data(upt_data)
    upt_city_data = upt_city_data.replace(0, np.nan).dropna(how='any', subset=year_cols)
    cities = list(upt_city_data.index.drop_duplicates())
    upt_data = upt_data[upt_data['UACE Code'].isin(cities)]
    # Separate reference tables with transit agency and city info
    agency_info = upt_data[other_cols].copy()
    agency_info.to_csv(paths.data / 'Agencies.csv')
    city_info = agency_info.drop(columns=['Agency Name']).drop_duplicates()
    # Merge list of individual cities in each urbanized area
    city_info['City State'] = city_info['City'].str.cat(city_info['State'], sep=' ')
    city_lists = city_info.groupby('UACE Code')['City State'].apply(lambda x: ', '.join(x))
    city_info = city_info.drop_duplicates(subset='UACE Code').set_index('UACE Code', drop=True)
    city_info['Cities'] = city_lists
    city_info = city_info.drop(columns=['City', 'State', 'City State'])
    city_info.to_csv(paths.data / 'Cities.csv')

    def parse_time_series(file, sheet, year_range=(1991, 2019), export=True,
                          per_capita=False, inflation_adjusted=False,
                          fname='', codes=cities):
        """
        Utility function to parse and export data from a time series table.

        Parameters
        ----------
        file : str
            Name of Excel spreadsheet file with time series data to import.
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
        fname : str [default: '']
            Name of output data file. If none provided, one will be generated
            based on the sheet name.
        codes : list of strings [default: ``cities``]
            List of UACE city codes to limit the data to.
        
        Returns
        -------
        train : pandas.DataFrame
            Years of training data per city.
        test : pandas.DataFrame
            Year(s) of test data per city.
        
        """
        years = list(range(year_range[0], year_range[1]+1))
        if fname == '':
            fname = sheet.replace(' ', '_') + '.csv'
        agency_data = read_time_series(
            file, 
            sheet,
            codes=codes,
            select_years=years
        )
        city_data = consolidate_city_data(agency_data)
        # Insert null data for missing years
        for i, year in enumerate(years):
            if str(year) not in city_data.columns:
                city_data.insert(i, str(year), np.nan * np.ones(city_data.shape[0]))
        if inflation_adjusted:
            city_data = adjust_for_inflation(city_data, inflation)
            fname = fname.replace('.csv', '_infladj.csv')
        if per_capita:
            city_data = normalize_population(city_data, population)
            fname = fname.replace('.csv', '_percap.csv')
        train, test = train_test_year_split(city_data)
        if export:
            train.to_csv(paths.data / 'train' / fname)
            test.to_csv(paths.data / 'test' / fname)
        return train, test

    def parse_time_series_fraction(file, partial_sheet, total_sheet, 
                                   year_range=(1991, 2019), export=True,
                                   fname=''):
        """
        Utility function to calculate funding or expenditure fractions.

        Parameters
        ----------
        file : str
            Name of Excel spreadsheet with time series data to import.
        partial_sheet : str
            Name of sheet in Excel file with partial (numerator) data.
        total_sheet : str or list
            Name(s) of sheet(s) in Excel file with total (denominator) data.
            If a list, data from all sheets will be summed.
        year_range : tuple of ints [default: (1991, 2019)]
            Start and end year (inclusive) of full data.
        export : bool [default: True]
            Whether to export the test and training data to separate CSVs.
        
        Returns
        -------
        train : pandas.DataFrame
            Years of training data per city.
        test : pandas.DataFrame
            Year(s) of test data per city.
        
        """
        partial_train, partial_test = parse_time_series(
            file, 
            partial_sheet, 
            year_range=year_range, 
            export=False, 
            per_capita=False, 
            inflation_adjusted=False
        )
        # The total can be the sum of multiple sheets, so we need to import
        # each one and sum them together.
        if isinstance(total_sheet, str):
            total_sheet = [total_sheet]
        all_train = []
        all_test = []
        for sheet in total_sheet:
            train, test = parse_time_series(
                file, 
                sheet, 
                year_range=year_range, 
                export=False, 
                per_capita=False, 
                inflation_adjusted=False
            )
            all_train.append(train)
            all_test.append(test)
        total_train = reduce(lambda x, y: x.add(y, fill_value=0), all_train)
        total_test = reduce(lambda x, y: x.add(y, fill_value=0), all_test)
        # Calculate fractions
        frac_train = partial_train / total_train
        frac_test = partial_test / total_test
        if export:
            if fname == '':
                fname = partial_sheet.replace(' ', '_') + '_frac.csv'
            frac_train.to_csv(paths.data / 'train' / fname)
            frac_test.to_csv(paths.data / 'test' / fname)
        return frac_train, frac_test

    #################################
    ### Vehicle & Ridership Stats ###
    #################################

    # Fare revenue
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'FARES', 
        per_capita=True, 
        inflation_adjusted=True
    )
    # Vehicles operated in maximum service (VOMS)
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VOMS', 
        per_capita=True
    )
    # Vehicle revenue miles (VRM)
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VRM', 
        per_capita=True
    )
    # Vehicle revenue hours (VRH)
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'VRH', 
        per_capita=True
    )
    # Unlinked passenger trips (UPT)
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'UPT', 
        per_capita=True
    )
    # Passenger Miles Traveled (PMT)
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'PMT', 
        per_capita=True
    )

    ##########################
    ### Operating Expenses ###
    ##########################

    # Total
    parse_time_series(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp Total', 
        per_capita=True, 
        inflation_adjusted=True
    )

    # Operating expenses - fraction for vehicle operations (VO)
    parse_time_series_fraction(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp VO',
        'OpExp Total'
    )

    # Operating expenses - fraction for vehicle maintenance (VM)
    parse_time_series_fraction(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp VM',
        'OpExp Total'
    )

    # Operating expenses - fraction for general administration (GA)
    parse_time_series_fraction(
        '2023 TS2.2 Service Data and Operating Expenses Time Series by System.xlsx', 
        'OpExp GA',
        'OpExp Total'
    )

    ######################################
    ### Operations and Capital Funding ###
    ######################################

    # Total operations funding per capita, inflation-adjusted
    parse_time_series(
        '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
        'Operating Total',
        per_capita=True,
        inflation_adjusted=True,
        fname='OpFund.csv'
    )

    # Capital funding per capita, inflation-adjusted
    parse_time_series(
        '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
        'Capital Total',
        per_capita=True,
        inflation_adjusted=True,
        fname='CapFund.csv'
    )

    # Fraction of total funding for operating expenses
    parse_time_series_fraction(
        '2023 TS1.2 Operating and Capital Funding Time Series.xlsx', 
        'Operating Total',
        ['Operating Total', 'Capital Total'],
        fname='OpFund_frac.csv'
    )

    ################################
    ### Funding Source Fractions ###
    ################################

    # Total funding per capita, inflation-adjusted
    parse_time_series(
        '2023 TS1.1 Total Funding Time Series_0.xlsx',
        'Total',
        per_capita=True,
        inflation_adjusted=True,
        fname='TotalFund.csv'
    )

    # Federal funding fraction
    parse_time_series_fraction(
        '2023 TS1.1 Total Funding Time Series_0.xlsx',
        'Federal',
        'Total',
        fname='FedFund_frac.csv'
    )

    # State funding fraction
    parse_time_series_fraction(
        '2023 TS1.1 Total Funding Time Series_0.xlsx',
        'State',
        'Total',
        fname='StateFund_frac.csv'
    )

    # Local funding fraction
    parse_time_series_fraction(
        '2023 TS1.1 Total Funding Time Series_0.xlsx',
        'Local',
        'Total',
        fname='LocalFund_frac.csv'
    )


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
    inflation_data = pd.read_csv(paths.data / 'Inflation.csv', index_col=0).T
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


if __name__ == '__main__':
    main()
