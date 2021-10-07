"""
Some definitions to be used in the scripts within this directory
"""
import numpy as np
import scipy.ndimage as ndim
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

bounds_lon = [-45,-35]
bounds_lat = [58,65]

datadir = '../Data/'
figdir = 'Methods_Figures/'

mooring_west = [-42.82545, 60.07013333]
mooring_east = [-30-31.765/60, 58+52.333/60]


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

    
    
def axSettingsLC(ax,bounds_lon,bounds_lat,eps=0.05):
    """
    axes settings for Lambert Conformal projection
    """
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor=(0.8,0.8,0.8))
    rect_lon=[bounds_lon[0]-eps, bounds_lon[1]+eps]
    rect_lat=[bounds_lat[0]-eps, bounds_lat[1]+eps]
    rect = mpath.Path([[rect_lon[0], rect_lat[0]],
                        [rect_lon[1], rect_lat[0]],
                        [rect_lon[1], rect_lat[1]],
                        [rect_lon[0], rect_lat[1]],
                        [rect_lon[0], rect_lat[0]],
                        ]).interpolated(20)
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_extent([bounds_lon[0], bounds_lon[1], bounds_lat[0] - 0.5, bounds_lat[1]])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), alpha=0)
    gl.xlocator = mticker.FixedLocator(np.arange(bounds_lon[0],bounds_lon[1]+0.1,1))
    gl.ylocator = mticker.FixedLocator(np.arange(bounds_lat[0],bounds_lat[1]+0.1,1))
    gl.xlabels_bottom = True
    gl.x_inline = False
    gl.xlabel_style = {'size': 12}
    gl.ylabels_left = True
    gl.ylabel_style = {'size': 12}