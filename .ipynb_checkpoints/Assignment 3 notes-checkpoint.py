import numpy as np
import pandas as pd

# Question 1 (20%)
def read_and_clean_energy_dataframe():
    # ToDo: verify existance of data file
    data_file = 'Energy Indicators.xls'
    
    # Create energy dataframe
    energy = pd.DataFrame()
    
    # take only the country records and drop the unwanted first two columns; rename the others
    energy = (pd.read_excel(data_file, usecols=[2, 3, 4,5], na_values=['...'])
                .iloc[16:243])
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

    # convert 'Energy Supply' from petajoules to gigajoules
    energy['Energy Supply'] = energy['Energy Supply'] * 1000000

    # rename countries
    # Note that several countries have superscripts for footnotes that were captured
    # in the initial read as trailing numbers in the country name
    countries_to_rename = {"Republic of Korea": "South Korea",
                           "United States of America": "United States",
                           "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                           "China, Hong Kong Special Administrative Region": "Hong Kong"}

    # First remove any footnote numbers and any trailing whitespace in the contry's name
    # then make the specific country name changes
    energy = (energy.replace(to_replace='^([^\d\(]+).*', value=r'\1', regex=True)
                    .replace(to_replace='\s*$', value='', regex=True)
                    .replace(to_replace={'Country' : countries_to_rename})
                    .set_index('Country'))
    return energy

def read_and_clean_GDP_dataframe():
    # ToDo: verify existance of data file
    data_file = 'world_bank.csv'
    
    # Create energy dataframe
    gdp = pd.DataFrame()
    
    # take only the country names and the last 10 years of data
    cols_to_use = ['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    gdp = (pd.read_csv(data_file, header=0, skiprows=[0, 1, 2, 3], usecols=cols_to_use)
             .rename(columns={'Country Name': 'Country'}))

    # Rename some countries to match the other data sets
    countries_to_rename = {"Korea, Rep.": "South Korea", 
                           "Iran, Islamic Rep.": "Iran",
                           "Hong Kong SAR, China": "Hong Kong"}
    gdp = (gdp.replace(to_replace={'Country' : countries_to_rename})
              .set_index('Country'))
    return gdp

def read_and_clean_ScimEn_dataframe():
    # ToDo: verify existance of data file
    data_file = 'scimagojr-3.xlsx'
    
    # Create scimen dataframe
    scimen = pd.DataFrame()
    
    # take only the country records and drop the unwanted first two columns; rename the others
    scimen = (pd.read_excel(data_file, na_values=['...'])
                .set_index('Country'))
    return scimen

def answer_one():
    # get the dataframes; all indexed to 
    energy = read_and_clean_energy_dataframe()
    GDP = read_and_clean_GDP_dataframe()
    ScimEn = read_and_clean_ScimEn_dataframe()
    
    # merge sequence to get columns in the requested order
    result = ScimEn.merge(energy, on='Country').merge(GDP, on='Country')

    return result.head(15)

answer_one()
# ----
# From the forums
# Assumes 'Country' is a column, not an index
﻿
energy.iloc[[0,-1]]
len(energy)  # 227
GDP.iloc[[0,-1]]
len(GDP)  # 264
ScimEn.iloc[[0,-1]]
len(ScimEn)  # 191

def test_energy(countries):
    """
    Input: a series/ the Country column in Energy
    utf-8 encoded i.e. when reading Energy use
    encoding='utf-8'
    """
    encodedC = '11,7,7,14,7,6,8,19,9,7,5,9,7,10,7,7,10,8,7,7,6,5,7,6,7,32,22,8,6,22,17,8,12,7,10,8,8,6,14,24,4,5,5,9,42,8,7,5,12,10,13,7,4,7,6,14,37,32,7,8,8,18,7,5,11,17,7,7,8,14,16,4,7,6,13,16,5,6,7,7,5,9,6,9,7,10,4,9,8,6,13,6,5,8,7,7,5,9,4,4,7,11,6,5,7,5,6,6,10,5,8,6,10,32,6,7,7,7,5,13,9,10,10,6,8,8,4,5,16,10,10,9,6,10,8,10,10,7,10,7,7,5,5,11,13,11,9,5,7,4,24,6,4,8,5,6,16,8,4,11,6,8,11,5,11,19,7,7,18,6,12,21,11,25,32,5,21,12,7,6,10,12,9,12,8,8,15,7,12,11,5,9,18,5,8,9,6,11,20,10,8,41,11,4,5,19,7,6,12,24,6,6,7,20,14,27,13,28,7,10,7,9,8,25,5,6,8'
    outcome = ['Failed\n', 'Passed\n']
    
    
    
    energy = pd.DataFrame()
    energy['original'] = pd.read_excel('Energy Indicators.xls',
                                   usecols=[1],encoding='utf-8',
                                   index_col=0).loc['Afghanistan':'Zimbabwe'].index.tolist()
    energy['tested'] = countries.str.len()
    energy['actual'] = encodedC.split(',')
    energy['actual'] = energy['actual'].astype(int)
    try:
        energy['Country'] = countries
    except Exception as e:
        print('Failed, error: ',e)
    
    res = 'Test number of records: '
    res += outcome[len(countries)==len(energy)]
    
    res += 'Test the column name: '
    res += outcome [countries.name == 'Country']
    
    res += 'Equality Test: '
    res += outcome[energy['tested'].equals(energy['actual'])]
    
    if not energy['tested'].equals(energy['actual']):
        res += '\nMismatched countries:\n'
        mismatch = energy.loc[energy['tested'] != (energy['actual']), [
            'original', 'Country', 'tested', 'actual']].values.tolist()
        res += '\n'.join('"{:}" miss-cleaned as  "{:}"'.format(o, r)
                         for o, r, s, v in mismatch)
    return res

# print(test_energy(get_energy().loc[:,'Country']))
print(test_energy(energy.loc[:,'Country']))

def test_gdp(countries):
    """
    Input: a series/ the Country column in GDP
    utf-8 encoded i.e. when reading GDP use
    encoding='utf-8'
    """
    encodedC = '5,7,11,6,7,10,20,9,7,14,19,9,7,10,7,7,5,12,10,8,7,12,22,7,6,7,7,6,8,17,6,8,24,6,30,11,15,5,5,13,8,11,8,7,10,10,22,4,7,14,6,14,7,8,8,7,18,7,43,26,19,45,21,7,16,9,7,5,7,8,14,40,7,4,6,13,21,5,14,7,5,9,6,11,13,17,6,7,9,9,4,6,11,9,8,38,7,5,7,9,16,9,9,9,8,11,5,14,7,4,4,7,6,5,7,6,5,10,5,15,8,8,19,11,6,6,49,7,7,7,5,9,25,44,10,13,9,19,19,7,25,9,10,6,16,24,7,6,7,10,8,26,6,16,13,14,4,5,7,50,10,8,24,10,10,9,6,8,13,7,13,5,7,9,11,6,5,5,11,12,4,18,8,6,4,11,5,16,6,24,11,25,8,8,27,25,16,5,7,18,6,10,12,5,7,9,15,12,11,10,7,6,42,11,18,12,21,8,15,8,6,9,25,10,20,24,4,42,44,4,8,10,12,52,49,11,5,23,41,19,7,6,6,8,6,7,19,7,13,10,30,13,22,21,7,7,18,5,5,11,12,16,6,8'
    outcome = ['Failed\n', 'Passed\n']
    
    
    
    GDP = pd.DataFrame()
    GDP['original'] = pd.read_csv('world_bank.csv',
                                  usecols=[0],encoding='utf-8',
                                  index_col=0).loc['Aruba':'Zimbabwe'].index.tolist()
    GDP['tested'] = countries.str.len()
    GDP['actual'] = encodedC.split(',')
    GDP['actual'] = GDP['actual'].astype(int)
    try:
        GDP['Country'] = countries
    except Exception as e:
        print('Failed, error: ',e)
    
    res = 'Test number of records: '
    res += outcome[len(countries)==len(GDP)]
    
    res += 'Test the column name: '
    res += outcome [countries.name == 'Country']
    
    res += 'Equality Test: '
    res += outcome[GDP['tested'].equals(GDP['actual'])]
    
    if not GDP['tested'].equals(GDP['actual']):
        res += '\nMismatched countries:\n'
        mismatch = GDP.loc[GDP['tested'] != (GDP['actual']), [
            'original', 'Country', 'tested', 'actual']].values.tolist()
        res += '\n'.join('"{:}" miss-cleaned as  "{:}"'.format(o, r)
                         for o, r, s, v in mismatch)
    return res
print(test_gdp(GDP['Country']))


"""
# Alternative merge strategy
# merge the first two, then the third in the requested order
merged2 = pd.merge(ScimEn, energy, how='inner', left_index=True, right_index=True)
merged3 = pd.merge(merged2, GDP, how='inner', left_index=True, right_index=True)

result = (ScimEn.merge(energy, on='Country')
                .merge(GDP, on='Country'))

energy[energy['Country'].str.contains('United')]
GDP[GDP['Country'].str.contains('United')]

sub_str = {'^([^\d\(]+).*' : r'\1'}
en2 = energy.iloc[232].replace(to_replace={'Country' : sub_str}, )
energy[energy['Country'].str.contains('United')].replace(to_replace={'Country' : sub_str})
energy[energy['Country'].replace(to_replace='^([^\d\(]+).*', value=r'\1', regex=True)

"""
# Question 2 (6.6%)
"""
The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?

This function should return a single number.
"""
def answer_two():
    # get the dataframes; all indexed to 'Country'
    energy = read_and_clean_energy_dataframe()
    GDP = read_and_clean_GDP_dataframe()
    ScimEn = read_and_clean_ScimEn_dataframe()
    
    # merge sequence to get columns in the requested order
    intersection = ScimEn.merge(energy, on='Country').merge(GDP, on='Country')
    union = ScimEn.merge(energy, on='Country', how='outer').merge(GDP, on='Country', how='outer')

    # return np.max([len(energy), len(GDP), len(ScimEn)]) - len(result) == incorrect answer
    return len(union) - len(intersection)

answer_two()


# Question 3 (6.6%)
"""
What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)

This function should return a Series named avgGDP with 15 countries and their average GDP sorted in descending order.
"""
def answer_three():

    def ave(row):
        # data = row[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
        data = row[np.r_[2006:2016].astype(str)]
        return pd.Series({'avgGDP': np.mean(data)})

    Top15 = answer_one()
    Top15['avgGDP'] = Top15.apply(ave, axis=1)
    Top15.sort_values('avgGDP', ascending=False, inplace=True)
    return Top15['avgGDP']

answer_three()

Top15.iloc[[0,-1]]

# Question 4 (6.6%)
"""
By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
From answer_three, 
This function should return a single number.
"""
def answer_four():
    Top15 = answer_one()
    # decade = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    decade = np.r_[2006:2016].astype(str)
    sixth_country = answer_three().index[5]
    sixth_gdp = Top15.loc[sixth_country][decade]
    return max(sixth_gdp) - min(sixth_gdp)

answer_four()

# Question 6 (6.6%)
"""
What country has the maximum % Renewable and what is the percentage?

This function should return a tuple with the name of the country and the percentage.
"""
def answer_six():
    Top15 = answer_one()
    cty = Top15['% Renewable'].idxmax()
    return tuple([cty, Top15.loc[cty]['% Renewable']])
# Too complicated
#    lst = (Top15.loc[Top15['% Renewable'] == Top15['% Renewable'].max()]['% Renewable'])
#    return tuple([lst.index, lst])

print(answer_six())
type(answer_six())

# Question 7 (6.6%)
"""
Create a new column that is the ratio of Self-Citations to Total Citations. What is the maximum value for this new column, and what country has the highest ratio?

This function should return a tuple with the name of the country and the ratio.
"""
def answer_seven():
    Top15 = answer_one()
    Top15['Citation Ratio'] = Top15['Self-citations'] / Top15['Citations']
    cnty = Top15['Citation Ratio'].idxmax()
    return tuple([cnty, Top15.loc[cnty]['Citation Ratio']])

answer_seven()

# Question 8 (6.6%)¶
"""
Create a column that estimates the population using Energy Supply and Energy Supply per capita. What is the third most populous country according to this estimate?

This function should return a single string value.
"""
def answer_eight():
    Top15 = answer_one()
    Top15['Population Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15.sort_values('Population Estimate', ascending=False, inplace=True)
    return Top15.iloc[2].name

answer_eight()

# Question 9 (6.6%)
"""
Create a column that estimates the number of citable documents per person. What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the .corr() method, (Pearson's correlation).

This function should return a single number.

(Optional: Use the built-in function plot9() to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)
"""
def answer_nine():
    Top15 = answer_one()
    Top15['Population Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citations per Capita'] = Top15['Citations'] / Top15['Population Estimate']
    return Top15['Citations per Capita'].corr(Top15['Energy Supply per Capita'])

answer_nine()

# Question 10 (6.6%)
"""
Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.

This function should return a series named HighRenew whose index is the country name sorted in ascending order of rank.
"""
def answer_ten():
    Top15 = answer_one()
    # Find the mean and create the boolean column converted to int
    Top15['HighRenew'] = (Top15['% Renewable'] >= Top15['% Renewable'].median()).astype(int)
    # sort df ascending by '% Renewable'
    Top15.sort_values('% Renewable', inplace=True)
    return Top15['HighRenew']

answer_ten()

#Question 11 (6.6%)¶
"""
Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.

ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}

This function should return a DataFrame with index named Continent ['Asia', 'Australia', 'Europe', 'North America', 'South America'] and columns ['size', 'sum', 'mean', 'std']
"""
def top15():
    # create base df
    Top15 = answer_one()
    # add continents
    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}
    # add continents
    Top15['Continent'] = pd.Series(ContinentDict, name='Continent')
    return Top15

def answer_eleven():
    # get the top 15 countries, with continents
    Top15 = top15()
    # estimate populations
    Top15['Population Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    # filter and reindex
    pop_stats = (Top15.filter(['Continent', '﻿Country', 'Population Estimate'])
                      .reset_index()
                      .set_index('Continent'))
    """ Terribly ugly solution, but it works """
    # group and get stats
    csize = pd.DataFrame(pop_stats.groupby('Continent')['Population Estimate'].size())
    csum  = pd.DataFrame(pop_stats.groupby('Continent')['Population Estimate'].sum())
    cmean = pd.DataFrame(pop_stats.groupby('Continent')['Population Estimate'].mean())
    cstd  = pd.DataFrame(pop_stats.groupby('Continent')['Population Estimate'].std())
    # tmp = (pop_stats.groupby('Continent', as_index=True)['Population Estimate'].size())
    # cstats = ({'size' : (csize, index=Continent),
    #            'sum'  : (csum,  index=Continent),
    #            'mean' : (cmean, index=Continent),
    #            'std'  : (csum,  index=Continent)})
    # cstats = ({'size' : (csize, index=csize.index),
    #            'sum'  : (csum,  index=csum.index),
    #            'mean' : (cmean, index=cmean.index),
    #            'std'  : (csum,  index=csum.index)})
    cstats = (csize.merge(csum, on='Continent')
                   .merge(cmean, on='Continent')
                   .merge(cstd, on='Continent'))
    cstats.columns = ['size', 'sum', 'mean', 'std']
    return cstats

answer_eleven()

# Question 12 (6.6%)
"""
Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?

This function should return a Series with a MultiIndex of Continent, then the bins for % Renewable. Do not include groups with no countries.
"""
def answer_twelve():
    # get the top 15 countries, with continents
    Top15 = top15()
    # reindex and filter
    Top15.reset_index(inplace=True)
    renew_bins = (Top15[['Continent', 'Country', '% Renewable']])
    return renew_bins.groupby(['Continent', pd.cut(renew_bins['% Renewable'], 5)])['Country'].size()

answer_twelve()

# Question 13 (6.6%)
"""
Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.

e.g. 317615384.61538464 -> 317,615,384.61538464

This function should return a Series PopEst whose index is the country name and whose values are the population estimate string.

Top15["PopEst"] = Top15["PopEst"].map(lambda x: "{:,}".format(x))
v= format(12345678, ',d')
"""
def answer_thirteen():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    return Top15["PopEst"].map(lambda x: "{:,}".format(x))

answer_thirteen()
