import os
os.chdir('/Users/dave/Workspaces/Python/Intro to Data Science with Python')
os.getcwd()

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import fileinput
import re

"""
Suppose we want to simulate the probability of flipping a fair coin 20 times, and getting a number greater than or equal to 15. Use np.random.binomial(n, p, size) to do 10000 simulations of flipping a fair coin 20 times, then see what proportion of the simulations are 15 or greater.
"""
trials = np.random.binomial(20, 0.5, 10000)
print((trials >= 15).mean())

states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


### Q1
def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt from https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) 
    list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    # compile the regexes
    state_re = re.compile('^([\w ]+)\[edit\]$')
    # region_re = re.compile('^([\w ,:"-]+) \(.*$')
    region_re = re.compile('^([^(]+) \(.*$')
    
    # process each line to extract and join state & region
    f = fileinput.input('university_towns.txt')
    df = pd.DataFrame(columns=["State", "RegionName"])
    row = 0
    for line in f:
        # first try to match the region, which is the most common match
        # the state always precedes the set of regions
        rv = region_re.match(line)
        if rv != None:
            cur_region = rv.group(1)
            df.loc[row] = [cur_state, cur_region]
            row += 1
        else:
            rv = state_re.match(line)
            if rv != None:
                cur_state = rv.group(1)
            else:  # these regions have no parenthetical trailer
                df.loc[row] = [cur_state, line]
                row += 1
    
    f.close()
    return df

get_list_of_university_towns().tail()


### Q2
# Read in GDP excel file [221:, 4:6], index = 'Qtr', col = 'GDP'
def get_gdp_df():

    # take only the records for second half of 1999 and beyond
    # some of 1999 is needed in case the recession starts 2000q1
    gdp = pd.read_excel('gdplev.xls', usecols=[4, 5], skiprows=217)
    gdp.columns = ['qtr', 'gdp']
    gdp.set_index('qtr', inplace=True)
    return gdp

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    data_file = 'gdplev.xls'
    
    gdp = get_gdp_df()
    
    # compare current quarter's gdp to previous and next quarters
    gdp['start'] = ((gdp['gdp'].shift() > gdp['gdp']) & 
                    (gdp['gdp'] > gdp['gdp'].shift(-1)))
    
    # return the first quarter that matches the start criteria
    return gdp[gdp['start']].iloc[0].name


### Q3
def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    
    # get gdp data after recession has started
    gdp = get_gdp_df().loc[get_recession_start():]
    
    # compare current quarter's gdp to previous and next quarters
    gdp['end'] = ((gdp['gdp'].shift() < gdp['gdp']) & 
                  (gdp['gdp'] < gdp['gdp'].shift(-1)))
    
    # return the first quarter that matches the end criteria
    # note that if the recession ends the quarter after 2 successive gdp increases
    # iloc arg will need to be incremented
    return gdp[gdp['end']].iloc[1].name


### q4
def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''

    # get gdp data during the recession
    gdp = get_gdp_df().loc[get_recession_start():get_recession_end()]
    
    return gdp['gdp'].idxmin()


### Q5
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    
    # Read in columns ['RegionName', 'State', '2000-01':] = [1, 2, 51:251]
    # and map state abbreviations to full state names
    housing = pd.read_csv('City_Zhvi_AllHomes.csv', 
                            header=0, 
                            usecols=np.r_[1, 2, 51:251], 
                            converters={'State':lambda x: states[x]})
    housing.set_index(['State',"RegionName"], inplace=True) 
    # Convert column headers from strings to datetimes
    housing.columns = pd.to_datetime(housing.columns, format='%Y-%m')
    # resample to get the quarterly mean
    housing = housing.resample('Q',axis=1).mean()
    # convert the column names back to strings
    housing = housing.rename(columns=lambda x: str(x.to_period('Q')).lower())
    
    return housing

housing = convert_housing_data_to_quarters()
housing.shape
housing.head()

# the statement following these comments is equivalent to these comments
# housing_data = pd.DataFrame()
# cols_to_use = np.r_[1, 2, 51:251]
# housing_data = pd.read_csv('City_Zhvi_AllHomes.csv', header=0, usecols=cols_to_use)
# housing_data['State'] = housing_data['State'].apply(lambda x: states[x])

# housing = pd.read_csv('City_Zhvi_AllHomes.csv', 
#                       header=0, 
#                       usecols=np.r_[1, 2, 51:251], 
#                       converters={'State':lambda x: states[x]})

### from forums
dates = pd.date_range('2000-3', '2016-9', freq='Q')
dates = dates.to_period('Q')
# I think the resample code goes here
dates = dates.map(lambda x: str(x).lower())
dates


### Q6
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    qtr_b4_recession = housing.columns.get_loc(get_recession_start()) - 1

    recession_housing = pd.DataFrame(columns=['price_ratio'])
    recession_housing['price_ratio'] = (housing.iloc[:, qtr_b4_recession] / 
                                        housing.loc[:, get_recession_bottom()])
    # hdf['PriceRatio'] = hdf[qrt_bfr_rec_start].div(hdf[rec_bottom])
    # recession_housing.head(10)

    # reformat the university towns df into a list of tuples 
    # for easy comparison with the housing data multi-index
    ut_tuples2 = get_list_of_university_towns().to_records(index=False).tolist()
    uni_towns_housing = recession_housing.loc[recession_housing.index.isin(ut_tuples2)]
    non_uni_towns_housing = recession_housing.loc[~recession_housing.index.isin(ut_tuples2)]
    # non-university town    10461
    # university town          269

    if float(uni_towns_housing.mean()) < float(non_uni_towns_housing.mean()):
        better = 'university town' 
    else:
        better = 'non-university town'
    # the index into the town type list is the result of the boolean test
    better = ('non-university town', 'university town')[float(uni_towns_housing.mean()) < float(non_uni_towns_housing.mean())]
    #                      price_ratio
    # town_type                       
    # non-university town     1.052376
    # university town         1.037718
    # diff                    0.014658

    p = ttest_ind(non_utowns, utowns, nan_policy='omit').pvalue
    
    return tuple(p < 0.01, p, better)

run_ttest()

#### from https://pandas.pydata.org/pandas-docs/stable/advanced.html
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
tuples = list(zip(*arrays))
tuples
# Out[3]: 
# [('bar', 'one'),
#  ('bar', 'two'), ...

# Or from forums
ut_tuples2 = ut.to_records(index=False).tolist()
uni_towns_housing = recession_housing.loc[ut_tuples2]
# or, to avoid generating NaNs for missing data
uni_towns_housing = recession_housing.loc[recession_housing.index.isin(ut_tuples2)]
non_uni_towns_housing = recession_housing.loc[-recession_housing.index.isin(ut_tuples2)]
# or using merge with a flag to indicate whether the index pairs are in both dfs
uni_towns = get_list_of_university_towns()
df = pd.merge(recession_housing.reset_index().
         uni_towns,
         on=uni_towns.columns.tolist(),
         indicator='_flag', how='outer')
uni_towns_housing = df[df['_flag']=='both']
non_uni_towns_housing = df[df['_flag']!='both']

uni_towns_housing = recession_housing.loc[recession_housing.index.isin(ut_tuples2)]
non_uni_towns_housing = recession_housing.loc[~recession_housing.index.isin(ut_tuples2)]
p2 = ttest_ind(non_uni_towns_housing, uni_towns_housing, nan_policy='omit').pvalue

# my first solution to Q6 is off slightly due to dropping Nans
    qtr_b4_recession = housing.columns.get_loc(get_recession_start()) - 1
    recession_housing = pd.DataFrame(columns=['price_ratio', 'town_type'])
    recession_housing['price_ratio'] = (housing.iloc[:, qtr_b4_recession] / 
                                        housing.loc[:, get_recession_bottom()])
    # hdf['PriceRatio'] = hdf[qrt_bfr_rec_start].div(hdf[rec_bottom])
    # recession_housing.head(10)

    # reformat the university towns df into a list of tuples 
    # for easy comparison with the housing data multi-index
    # also see notes from forums below
    ut_tuples = list(get_list_of_university_towns().itertuples(index=False, name=None))
    # the index into the town type list is the result of the boolean test
    recession_housing['town_type'] = recession_housing.apply(lambda row: 
                                                             ('non-university town', 'university town')[row.name in ut_tuples], 
                                                             axis=1)
    # recession_housing['town type'].value_counts()
    # non-university town    10461
    # university town          269

    town_means = recession_housing.groupby('town_type').mean()
    better = town_means['price_ratio'].idxmin()
    #                      price_ratio
    # town_type                       
    # non-university town     1.052376
    # university town         1.037718
    # diff                    0.014658

    p = ttest_ind(non_utowns, utowns, nan_policy='omit').pvalue
    
    return tuple(p < 0.01, p, better)
