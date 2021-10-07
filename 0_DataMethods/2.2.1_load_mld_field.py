"""
We load the mixed layer depth (MLD) for 1993-2019 and compute the mean field
for months in which the maximum MLD exceeds 1000 m.
"""
import numpy as np
import scipy.io as sio
import netCDF4 as nc
import datetime
from datetime import timedelta
from info import bounds_lat, bounds_lon, datadir

# Load the data set
fn =  '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/global-reanalysis-phy-001-030-monthly_MLD.nc'
ds = nc.Dataset(fn)

time = ds['time'][:]
t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
dates = np.empty(len(time), dtype=object)
for i in range(len(dates)):
    dates[i] = t0 + timedelta(hours=time[i])
lat = ds['latitude'][:]
lon = ds['longitude'][:]

# Select area of interest
ind_lat_lower = np.argwhere(lat==bounds_lat[0])[0][0]
ind_lat_upper = np.argwhere(lat==bounds_lat[1])[0][0]
ind_lon_left = np.argwhere(lon==bounds_lon[0])[0][0]
ind_lon_right = np.argwhere(lon==bounds_lon[1])[0][0]

# Load MLD data in area of interest
mld = ds['mlotst'][:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]
mld = mld[:,::-1,:] # reverse lat axis (same as for velocity fields)

# Select months in which the max MLD within the area was more than 1000 m
large_mld_months = []
for i in range(len(dates)):
    if np.max(mld[i,:,:]) > 1000:
        large_mld_months.append(i)

# Save indices of months with large MLD
np.save(datadir+'large_mld_months', large_mld_months)

# Average over months with max MLD > 1000 m
large_mld_mean = np.ma.mean([mld[j,:,:] for j in large_mld_months],axis=0)
large_mld_mean = np.ma.filled(large_mld_mean, np.nan) # replace masked values with NaNs

# Save averaged MLD field
large_mld_field = {"mld_field": large_mld_mean}
sio.savemat(datadir+"mld_field.mat", large_mld_field)
