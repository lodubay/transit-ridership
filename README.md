# transit-ridership

A project for the Erdős Institute's Fall 2024 Data Science Boot Camp.

## Overview

Many cities across the U.S. are increasing investment in public transportation in order
to reduce greenhouse gas emissions and improve transportation safety. Public transit is
getting increasingly expensive, and large transit projects can cost many billions of
dollars. To build a more effective transportation system with limited funding, it is
important to understand the factors that have the largest effect on ridership. We develop
two models to predict public transit ridership per capita for 182 cities based on a
number of factors, such as vehicle statistics and funding allocation, from the U.S. 
Department of Transportation's [National Transit Database](https://www.transit.dot.gov/ntd). 
The first is a bespoke model trained on 28 years of data for each city, while the second 
is trained on most city data at once.

## Repository Structure

[Read the executive summary](Executive_Summary.pdf) and [see the presentation
slides](Transit_Ridership_Slides.pdf)

### The `src/` directory

- The [`parse_data.py`](src/parse_data.py) script imports time-series data from the Excel
spreadsheets and outputs the parsed and adjusted training and testing data.
- The [`EDA.ipynb`](src/EDA.ipynb) notebook contains exploratory data analysis.
- The [`City_Specific_Model.ipynb`](src/City_Specific_Model.ipynb) notebook fits 
a regression for each city independently.
- The [`all_city_model.ipynb`](src/all_city_model.ipynb) notebook fits a single 
XGBoost regression model for all cities.

### The `data/` directory

- Excel time-series spreadsheets downloaded from the [National Transit Database
](https://www.transit.dot.gov/ntd) are located in `data/time_series`. 
- Census data are located in `data/census`.
- Parsed data are split between the `data/train` and `data/test` folders. The
former contains adjusted data for all cities through 2018, and the latter
contains only 2019 data. Each feature has its own CSV file.
- The `data/Inflation.csv` file contains the value of a 2019 dollar for each
year for which we have data, taken from the [U.S. Bureau of Labor Statistics
Inflation Calculator](https://data.bls.gov/cgi-bin/cpicalc.pl).
- Additional info on each city and transit agencies is stored in `data/Cities.csv`
and `data/Agencies.csv`, respectively.
