import pandas as pd
import numpy as np

from pytools.utils import get_datetime

###############################################################################

def read_ecosan(fname):
    """
        Read ECOSAN project ascii file (currentmeter) and return a pandas Dataframe.
    """
    df = pd.read_csv(fname, skiprows=11, sep=',', header=None)
    
    # get datetime
    dt = get_datetime(df[5])
    df = df.drop(5, axis=1)
    df.index = pd.to_datetime(dt)

    # create header 
    header = pd.read_csv(fname, nrows=1, skiprows=10).columns[0].split(' ')
    header = [head for head in header if head != '' ]
    header.remove('TIME')
    header.remove('DATE')
    
    df.columns = header
    
    return df
    
###############################################################################

def read_raw_piof(fname):
    """
        Read PIOF project raw files, returning a pandas DataFrame with _metadata within.
    """
    f = open(fname, 'r')

    dep = []
    temp= []
    salt= []
    meta= {}

    for row in f.readlines():
        # get data
        if row[0] == '-':
            columns = row.split(',')
            dep.append(float(columns[0])*(-1))
            temp.append(float(columns[1]))
            salt.append(float(columns[2].replace('\n', '')))

        # get coordinates
        elif row[0] == 'l':
            _ = row.split(',')
            meta['lon'] = (-1)*float(_[0].split(',')[0].split('=')[1])
            meta['lat'] = (-1)*float(_[1].split(',')[0].split('=')[1])

        elif row[0] == ' ':
            _ = row.split('=')[1].split('/')
            yyyy = int(float(_[0]) + 1000)
            mm = int(float(_[1]))
            dd = int(float(_[2]))
            hh = int(float(_[3]))

            meta['datetime'] = pd.to_datetime(f'{yyyy}{mm}{dd}{hh}', format='%Y%m%d%H')

    df = pd.DataFrame({'temp': temp, 'salt': salt}, index=dep)

    df._metadata = meta
    
    return df