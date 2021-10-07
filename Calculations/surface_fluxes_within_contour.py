"""
Computing ERA5 atmospheric fluxes over the DCA
"""
import scipy.io as sio
import netCDF4 as nc
import numpy as np
from info import datadir#, print_header


#%%
# =============================================================================
# Load DCA area data
# =============================================================================

mask = np.load(datadir+'DCA_mask_ERA5.npy')

x = np.load(datadir+'DCA_contour_x.npy')
y = np.load(datadir+'DCA_contour_y.npy')
bounds_lat = [np.min(y), np.max(y)]
bounds_lon = [np.min(x), np.max(x)]

#%%
# =============================================================================
# Load ERA5 data
# =============================================================================

fn = datadir+'ERA5_1993-2019_monthly_mean.nc'
ds = nc.Dataset(fn)
    
time = ds['time'][:]
lat = ds['latitude'][:]
lon = ds['longitude'][:]

E = ds['e'][:] # evaporation
P = ds['tp'][:] # total precipitation
E_P = E+P # net freshwater flux

slhf = ds['slhf'][:] # surface upward latent heat flux
ssr = ds['ssr'][:] # surface net downward shortwave flux (solar radiation)
strr = ds['str'][:] # surface net upward longwave flux (thermal radiation)
sshf = ds['sshf'][:] # surface upward sensible heat flux
Q = slhf+ssr+strr+sshf # total heat flux

# Select area of interest
ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]

lat = lat[ind_lat_upper:ind_lat_lower+1] 
lon = lon[ind_lon_left:ind_lon_right+1]

E = E[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
P = P[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
E_P = E_P[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
Q = Q[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]

# slhf = slhf[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
# ssr = ssr[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
# strr = strr[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]
# sshf = sshf[:,ind_lat_upper:ind_lat_lower+1,ind_lon_left:ind_lon_right+1]



#%%
def timeseries_in_contour(data,bounds_lat,bounds_lon,contour_mask):
    data_new = np.ma.filled(data,np.nan)
    data_masked = np.ma.zeros(np.shape(data_new))
    for i in range(len(time)):
        data_masked[i,:,:] = np.ma.masked_array(data_new[i,:,:],mask)
    
    weights=np.cos(lat/180*np.pi) # weights for latitude
    data_area = np.ma.zeros(len(time))
    for i in range(len(time)):
        data_area[i] = np.ma.average(np.ma.average(data_masked[i,:,:],axis=0,weights=weights))
    data_area = np.ma.filled(data_area,np.nan)
    
    return data_area



E_DCA = timeseries_in_contour(E,bounds_lat,bounds_lon,mask)
P_DCA = timeseries_in_contour(P,bounds_lat,bounds_lon,mask)
E_P_DCA = timeseries_in_contour(E_P,bounds_lat,bounds_lon,mask)
Q_DCA = timeseries_in_contour(Q,bounds_lat,bounds_lon,mask)

# slhf_DCA = timeseries_in_contour(slhf,bounds_lat,bounds_lon,mask)
# ssr_DCA = timeseries_in_contour(ssr,bounds_lat,bounds_lon,mask)
# strr_DCA = timeseries_in_contour(strr,bounds_lat,bounds_lon,mask)
# sshf_DCA = timeseries_in_contour(sshf,bounds_lat,bounds_lon,mask)

#%%
# Save data in matfile
data = {"time": time, "E": E_DCA, "P": P_DCA, "E_P": E_P_DCA, "Q": Q_DCA}
sio.savemat(datadir+"surface_fluxes_DCA.mat", data)

# data = {"time": time, "slhf": slhf_DCA, "ssr": ssr_DCA, "strr": strr_DCA, "sshf": sshf_DCA}
# sio.savemat(datadir+"separate_heat_fluxes.mat",data)

