import os
from dateutil.parser import parse
import pandas as pd
import numpy as np

###############################################################################

def get_datetime(s):
    s = pd.DataFrame(s)
    datetime_list = []
    for i,row in s.iterrows():
        datetime_list.append(parse(row.values[0]))
        
    return np.asarray(datetime_list)

###############################################################################

# function to reduce the linewidth between contourfs
def configuring_for_pdf(cf):

    for c in cf.collections:
        c.set_edgecolor('face')
        c.set_linewidth(0.00000000001)

###############################################################################
