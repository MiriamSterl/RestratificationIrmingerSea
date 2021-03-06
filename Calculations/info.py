"""
Some definitions to be used in the scripts within this directory
"""
import numpy as np
import scipy.ndimage as ndim
import datetime

bounds_lon = [-45,-35]
bounds_lat = [58,65]

mooring_west = [-42.82545, 60.07013333]
mooring_east = [-30-31.765/60, 58+52.333/60]

datadir = '../Data/'

def getMonths(dates):
    months = np.empty(12, dtype=object)
    for j in range(months.shape[0]):
        months[j] = [] # initialize as empty lists
    for i in range(len(dates)):
        m = dates[i].month
        months[m-1].append(i)
    return months