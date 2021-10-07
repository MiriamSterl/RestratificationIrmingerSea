"""
Some definitions to be used in the scripts within this directory
"""
import numpy as np
import scipy.ndimage as ndim
import datetime

bounds_lon = [-45,-35]
bounds_lat = [58,65]

datadir = '../Data/'
figdir = 'Interpretation_Figures/'


def smooth_criminal(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    
    Source: 
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndim.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndim.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss


def print_header(string):
    print()
    print("##################################################################")
    print(string)
    print("##################################################################")
    print()
    

def getMonths(dates):
    months = np.empty(12, dtype=object)
    for j in range(months.shape[0]):
        months[j] = [] # initialize as empty lists
    for i in range(len(dates)):
        m = dates[i].month
        months[m-1].append(i)
    return months


def makeClimatology(data,dates):
    """
    Computes the climatology of a timeseries of depth profiles.
    
    Parameters:
        data (numpy array): contains the timeseries of depth profiles,
                            must have dimensions (time,depth).
    """
    months = getMonths(dates)
    nvert = np.shape(data)[1] # number of vertical levels
    data_clim = np.zeros((12,nvert))
    for i in range(12):
        data_clim[i,:] = np.nanmean([data[j,:] for j in months[i]],axis=0)
    return data_clim