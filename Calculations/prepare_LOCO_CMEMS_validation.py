"""
Prepare data from LOCO mooring for data validation
+ data from CMEMS at LOCO location
"""
import scipy.io as sio
import numpy as np
import netCDF4 as nc
import gsw
import iris
import iris.analysis
import xarray as xr
import datetime
import pandas as pd
from scipy import interpolate
from info import datadir


#%%
# =============================================================================
# Load LOCO coordinates
# =============================================================================
loco = sio.loadmat(datadir+'LOCO2x.mat')['LOCO2']

lat_loco = loco['lat'][0,0][0]
lon_loco = loco['lon'][0,0][0]



#%% CMEMS
# =============================================================================
# Load CMEMS data
# =============================================================================

# 4 grid points around LOCO mooring
bounds_lat = [59+2/12,59+3/12]
bounds_lon = [-39-7/12,-39-6/12]

#%
# =============================================================================
# Load time, depth, salinity and temperature
# =============================================================================
loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'

def load_TS_CMEMS(years,incl_coord=False):
    fn = loadpath + 'global-reanalysis-phy-001-030-monthly_'+years+'.nc'
    ds = nc.Dataset(fn)
    
    time = ds['time'][:]
    depth = ds['depth'][:]
    depth_2000_ind = np.where(depth>2000)[0][0]
    depth_cmems = depth[0:depth_2000_ind+1]
    lat = ds['latitude'][:]
    lon = ds['longitude'][:]
    
    # Select area of interest
    ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
    ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
    ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
    ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]
    lat = lat[ind_lat_lower:ind_lat_upper+1] 
    lon = lon[ind_lon_left:ind_lon_right+1]
    
    SP = ds['so'][:,0:depth_2000_ind+1,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1] # practical salinity
    PT = ds['thetao'][:,0:depth_2000_ind+1,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1] # potential temperature
    
    if incl_coord:
        return time, SP, PT, depth_cmems, lat, lon
    else:
        return time, SP, PT
    
decades = ['2003-2007', '2008-2012', '2013-2017']
time1, SP1, PT1 = load_TS_CMEMS(decades[0])
time2, SP2, PT2 = load_TS_CMEMS(decades[1])
time3, SP3, PT3, depth_cmems, lat, lon = load_TS_CMEMS(decades[2],True)

time_val = np.hstack((time1,time2,time3))
t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
dates_val = np.array([t0 + datetime.timedelta(hours=i) for i in time_val])

SP = np.vstack((SP1,SP2,SP3))
PT = np.vstack((PT1,PT2,PT3))

del SP1,SP2,SP3,PT1,PT2,PT3


# =============================================================================
# Load CMEMS MLD data
# =============================================================================
fn = loadpath + 'global-reanalysis-phy-001-030-monthly_MLD.nc'
ds = nc.Dataset(fn)
time_mld = ds['time'][:]
t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
dates_mld = np.array([t0 + datetime.timedelta(hours=i) for i in time_mld])
lat = ds['latitude'][:]
lon = ds['longitude'][:]
ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]
lat = lat[ind_lat_lower:ind_lat_upper+1] 
lon = lon[ind_lon_left:ind_lon_right+1]

mld_cmems_raw = ds['mlotst'][120:300,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]



# =============================================================================
# Interpolate CMEMS data to mooring location, then compute SA and CT there
# =============================================================================

SA_cmems_interp = np.zeros((len(time_val),len(depth_cmems)))
CT_cmems_interp = np.zeros((len(time_val),len(depth_cmems)))
mld_cmems = np.zeros((len(time_val)))

pos_loco = [('latitude', lat_loco), ('longitude', lon_loco)]
for i in range(len(time_val)):
    mld_iris = xr.DataArray(data=mld_cmems_raw[i,:,:], dims=["latitude", "longitude"], coords=[lat,lon]).to_iris()
    interp_mld = mld_iris.interpolate(pos_loco, iris.analysis.Linear())
    mld_cmems[i] = xr.DataArray.from_iris(interp_mld).data
    
    for j in range(len(depth_cmems)):
        SP_iris = xr.DataArray(data=SP[i,j,:,:], dims=["latitude", "longitude"], coords=[lat,lon]).to_iris()
        PT_iris = xr.DataArray(data=PT[i,j,:,:], dims=["latitude", "longitude"], coords=[lat,lon]).to_iris()
        interp_SP = SP_iris.interpolate(pos_loco, iris.analysis.Linear())
        interp_PT = PT_iris.interpolate(pos_loco, iris.analysis.Linear())
        SP_interp_loco = xr.DataArray.from_iris(interp_SP).data
        PT_interp_loco = xr.DataArray.from_iris(interp_PT).data
        
        p = gsw.p_from_z(depth_cmems[j]*-1,lat_loco)
        SA_cmems_interp[i,j] = gsw.SA_from_SP(SP_interp_loco,p,lon_loco,lat_loco)
        CT_cmems_interp[i,j] = gsw.CT_from_pt(SA_cmems_interp[i,j],PT_interp_loco)

SA_cmems_interp = np.transpose(SA_cmems_interp)
CT_cmems_interp = np.transpose(CT_cmems_interp)




#%% LOCO
# =============================================================================
# Load LOCO data
# =============================================================================

time_loco = loco['mtime'][0,0][0]
dates_loco = pd.to_datetime(time_loco-719529,unit='d').round('s')
dates_val_loco = dates_loco

p_loco = loco['P'][0,0][0]
depth_loco = gsw.z_from_p(p_loco,lat_loco)*-1
depth_ind = np.argwhere(depth_loco>=depth_cmems[-1])[0][0]
depth_val_loco = depth_loco[0:depth_ind+1]
Time_loco, Depth_loco = np.meshgrid(time_val,depth_val_loco)

SA_loco_raw = loco['SA'][0,0][0:depth_ind+1,:]
CT_loco_raw = loco['CT'][0,0][0:depth_ind+1,:]

# =============================================================================
# MLD
# =============================================================================
data_loco_mld = sio.loadmat(datadir+'LOCO_MLD.mat')
mld_loco_raw = np.zeros(len(time_loco))
mld_loco_raw[0:-1] = data_loco_mld['MLD'][0]
mld_loco_raw[-1] = np.nan


#%%
# =============================================================================
# Transform LOCO data to monthly data
# =============================================================================

def daily_to_monthly(depth_level,data):
    j = 0
    d = depth_level
    data_monthly = np.zeros((len(dates_val)))#*np.nan
    for i in range(len(dates_val)):
        date = dates_val[i]
        data_num = 0
        data_sum = 0
        while(dates_val_loco[j].year==date.year and dates_val_loco[j].month==date.month and j+1<len(dates_val_loco)):
            if not np.isnan(data[d,j]):
                data_sum += data[d,j]
                data_num += 1
            j += 1
        if data_num>0:
            data_monthly[i] = data_sum/data_num
        else:
            data_monthly[i] = np.nan
    return data_monthly

SA_loco = np.zeros((len(depth_val_loco),len(dates_val)))
CT_loco = np.zeros((len(depth_val_loco),len(dates_val)))

for d in range(len(depth_val_loco)):
    SA_loco[d,:] = daily_to_monthly(d,SA_loco_raw)
    CT_loco[d,:] = daily_to_monthly(d,CT_loco_raw)
    
    


mld_loco = np.zeros((len(dates_val)))
j = 0
for i in range(len(dates_val)):
    date = dates_val[i]
    mld_num = 0
    mld_sum = 0
    while(dates_val_loco[j].year==date.year and dates_val_loco[j].month==date.month and j+1<len(dates_val_loco)):
        if not np.isnan(mld_loco_raw[j]):
            mld_sum += mld_loco_raw[j]
            mld_num += 1
        j += 1
    if mld_num > 0:
        mld_loco[i] = mld_sum/mld_num
    else:
        mld_loco[i] = np.nan


#%%
# =============================================================================
# Regrid CMEMS data to regular depth levels
# =============================================================================


def regrid_vertical(data, time, vert):
    """
    vert: either depth or pressure
    """
    f = interpolate.interp2d(time,vert,data, kind='linear')
    xi = time
    yi = np.linspace(vert[0],vert[-1],len(vert))
    Time,Vert = np.meshgrid(xi,yi)
    data_interp = f(xi,yi)
    return Time,Vert,data_interp

_,_,SA_cmems = regrid_vertical(SA_cmems_interp, time_val, depth_cmems)
Time_cmems,Depth_cmems,CT_cmems = regrid_vertical(CT_cmems_interp, time_val, depth_cmems)


#%% Density
# =============================================================================
# Compute density profiles
# =============================================================================

sigma0_cmems = gsw.sigma0(SA_cmems,CT_cmems)
sigma0_loco = gsw.sigma0(SA_loco,CT_loco)

    
#%% Save data
# =============================================================================
# Save CMEMS and LOCO results
# =============================================================================

data = {"time": time_val, "depth_cmems": depth_cmems, "depth_loco": depth_val_loco,
        "Time_cmems": Time_cmems, "Time_loco": Time_loco,
        "Depth_cmems": Depth_cmems, "Depth_loco": Depth_loco,
        "SA_cmems_depth": SA_cmems_interp, "CT_cmems_depth": CT_cmems_interp,
        "SA_cmems": SA_cmems, "SA_loco": SA_loco,
        "CT_cmems": CT_cmems, "CT_loco": CT_loco,
        "sigma0_cmems": sigma0_cmems, "sigma0_loco": sigma0_loco,
        "mld_cmems": mld_cmems, "mld_loco": mld_loco}
sio.savemat(datadir+"LOCO_CMEMS_val.mat",data)


