import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import fileinput
import re

import os
os.chdir('/Users/dave/Workspaces/Python/Intro to Data Science with Python')


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
    region_re = re.compile('^([\w ]+) \(.*$')
    region_num = -1  # initial increment will take this to 0

    # process each line to extraxt and join state & region
    df = pd.DataFrame(columns=["State", "RegionName"])
    f = fileinput.input('tmp_ut.txt')
    for line in f:
        # reset region
        cur_region = ''

        # first try to match the region, which is the most common match
        # the state always precedes the set of regions
        rv = region_re.match(line)
        if rv != None:
            region_num += 1
            cur_region = rv.group(1)
            df.loc[region_num] = [cur_state, cur_region]
        else:
            rv = state_re.match(line)
            if rv != None:
                cur_state = rv.group(1)

    f.close()
    return df
print(df.head())