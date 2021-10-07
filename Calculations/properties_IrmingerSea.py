"""
Computing velocities, salinity, temperature, eddy kinetic energy and advection 
until a depth of ~500 m (depth-averaged) in the Irminger Sea
"""
import scipy.io as sio
import netCDF4 as nc
import gsw
import numpy as np
import datetime
#from info import datadir


#%%
# =============================================================================
# LOAD DATA
# =============================================================================


# loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'
loadpath = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/'

def load_uvST(years, return_coord = False):
    """
    Load CMEMS velocities, potential temperature and practical salinity;
    compute conservative temperature and absolute salinity.
    
    Parameters:
        years : year range as given in the filename of the .nc file
        return_coord : whether or not to return time, depth, lat, lon
    """
    fn = loadpath + 'global-reanalysis-phy-001-030-monthly_'+years+'.nc'
    ds = nc.Dataset(fn)
    
    time = ds['time'][:]
    depth = ds['depth'][:]
    lat = ds['latitude'][:]
    lon = ds['longitude'][:]
    
    maxdepth = np.where(depth>500)[0][0]
    depth = depth[0:maxdepth+1]
    
    # Load velocities
    uo = ds['uo'][:,0:maxdepth+1,:,:] # zonal velocity
    vo = ds['vo'][:,0:maxdepth+1,:,:] # meridional velocity

    # Load potential temperature and practical salinity
    PT = ds['thetao'][:,0:maxdepth+1,:,:] # potential temperature
    SP = ds['so'][:,0:maxdepth+1,:,:] # practical salinity
    
    # Compute absolute salinity, conservative temperature and potential density
    SA = np.ma.zeros(np.shape(SP)) # absolute salinity 
    CT = np.ma.zeros(np.shape(PT)) # conservative temperature
    p = np.zeros((len(depth),len(lat))) # sea pressure
    for i in range(len(time)):
        for j in range(len(depth)):
            for k in range(len(lat)):
                p[j,k] = gsw.p_from_z(depth[j]*-1, lat[k])
                for l in range(len(lon)):
                    SA[i,j,k,l] = gsw.SA_from_SP(SP[i,j,k,l], p[j,k], lat[k], lon[l])
                    CT[i,j,k,l] = gsw.CT_from_pt(SA[i,j,k,l], PT[i,j,k,l])
                    
    if return_coord:
        return uo, vo, SA, CT, time, depth, lat, lon
    else:
        return uo, vo, SA, CT, time


# =============================================================================
# Load data for each decade
# =============================================================================

decades = ['1993-1996', '1997-2002', '2003-2007', '2008-2012', '2013-2017', '2018-2019']

uo1, vo1, SA1, CT1, time1 = load_uvST(decades[0])
uo2, vo2, SA2, CT2, time2 = load_uvST(decades[1])
uo3, vo3, SA3, CT3, time3 = load_uvST(decades[2])
uo4, vo4, SA4, CT4, time4 = load_uvST(decades[3])
uo5, vo5, SA5, CT5, time5 = load_uvST(decades[4])
uo6, vo6, SA6, CT6, time6, depth, lat, lon = load_uvST(decades[5], return_coord=True)

# Combine data
time = np.hstack((time1,time2,time3,time4,time5,time6))
del time1,time2,time3,time4,time5,time6
uo = np.ma.vstack((uo1,uo2,uo3,uo4,uo5,uo6))
del uo1,uo2,uo3,uo4,uo5,uo6
vo = np.ma.vstack((vo1,vo2,vo3,vo4,vo5,vo6))
del vo1,vo2,vo3,vo4,vo5,vo6
SA = np.ma.vstack((SA1,SA2,SA3,SA4,SA5,SA6))
del SA1,SA2,SA3,SA4,SA5,SA6
CT = np.ma.vstack((CT1,CT2,CT3,CT4,CT5,CT6))
del CT1,CT2,CT3,CT4,CT5,CT6


#%%
# =============================================================================
# WHOLE TIME SERIES
# =============================================================================

def compMeanAndAnomaly(data):
    """
    Computes the mean and anomaly per grid point over the whole time series
    """
    data_mean = np.ma.mean(data,axis=0)
    data_anom = data - data_mean
    return data_mean, data_anom


u_mean, u_anom = compMeanAndAnomaly(uo)
v_mean, v_anom = compMeanAndAnomaly(vo)
S_mean, S_anom = compMeanAndAnomaly(SA)
T_mean, T_anom = compMeanAndAnomaly(CT)
EKE = (1/2)*((u_anom)**2+(v_anom)**2)
EKE_mean, EKE_anom = compMeanAndAnomaly(EKE)



#%%
# =============================================================================
# CLIMATOLOGY
# =============================================================================

# Transform time to dates and sort by month
t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
dates = np.array([t0 + datetime.timedelta(hours=i) for i in time])
 
# Find which index corresponds with which month    
months = np.empty(12, dtype=object)
for j in range(months.shape[0]):
    months[j] = [] # initialize as empty lists
for i in range(len(dates)):
    m = dates[i].month
    months[m-1].append(i)

def compMeanAndAnomalyMonth(data):
    """
    Computes the mean and anomaly per grid point per calendar month in the time series
    """
    data_clim = np.empty(12,dtype=object)
    for i in range(12):
        data_clim[i] = np.ma.mean([data[j,:,:] for j in months[i]],axis=0)
    data_clim_mean = np.ma.vstack([np.expand_dims(data_clim[j],0) for j in range(12)])
    data_clim_anom = np.ma.zeros(np.shape(data))
    for t in range(len(time)):
        m = dates[t].month - 1
        data_clim_anom[t,:,:,:] = data[t,:,:,:] - data_clim_mean[m,:,:,:]
    return data_clim_mean, data_clim_anom


u_mean_month, u_anom_month = compMeanAndAnomalyMonth(uo)
v_mean_month, v_anom_month = compMeanAndAnomalyMonth(vo)
S_mean_month, S_anom_month = compMeanAndAnomalyMonth(SA)
T_mean_month, T_anom_month = compMeanAndAnomalyMonth(CT)
#EKE_month = (1/2)*((u_anom_month)**2+(v_anom_month)**2)
EKE_mean_month, EKE_anom_month = compMeanAndAnomalyMonth(EKE)


#%%
# =============================================================================
# WEIGHTED AVERAGE OVER DEPTH (averaging done in next step)
# =============================================================================

# Load layer thickness data
fn = loadpath + 'GLO-MFC_001_024_coordinates.nc'
ds = nc.Dataset(fn)

bounds_lon = [-45,-25]
bounds_lat = [55,65]
lon2 = ds['longitude'][:]
lat2 = ds['latitude'][:]
ind_lat_lower = np.argwhere(lat2>=bounds_lat[0])[0][0]
ind_lat_upper = np.argwhere(lat2<=bounds_lat[1])[-1][0]
ind_lon_left = np.argwhere(lon2>=bounds_lon[0])[0][0]
ind_lon_right = np.argwhere(lon2<=bounds_lon[1])[-1][0]

e3t = ds['e3t'][0:len(depth),ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]

def depthAverage(data):
    firstdim = np.shape(data)[0]
    if firstdim == len(depth):
        data_1depth = np.ma.filled(np.ma.average(data,weights=e3t,axis=0),np.nan)
    else:
        data_1depth = np.ma.zeros((firstdim,len(lat),len(lon)))
        for i in range(firstdim):
            data_1depth[i,:,:] = np.ma.average(data[i,:,:,:], weights=e3t, axis=0)
        data_1depth = np.ma.filled(data_1depth,np.nan)
    return data_1depth


#%%
# =============================================================================
# SAVE DATA
# =============================================================================

coords = {"time": time, "depth": depth, "lat": lat, "lon": lon}

properties_mean = {"u": depthAverage(u_mean), "v": depthAverage(v_mean), 
                   "S": depthAverage(S_mean), "T": depthAverage(T_mean),
                   "EKE": depthAverage(EKE_mean)}
properties_anom = {"u": depthAverage(u_anom), "v": depthAverage(v_anom), 
                   "S": depthAverage(S_anom), "T": depthAverage(T_anom),
                   "EKE": depthAverage(EKE_anom)}
properties_mean_month = {"u": depthAverage(u_mean_month), "v": depthAverage(v_mean_month), 
                   "S": depthAverage(S_mean_month), "T": depthAverage(T_mean_month),
                   "EKE": depthAverage(EKE_mean_month)}
properties_anom_month = {"u": depthAverage(u_anom_month), "v": depthAverage(v_anom_month), 
                   "S": depthAverage(S_anom_month), "T": depthAverage(T_anom_month),
                   "EKE": depthAverage(EKE_anom_month)}

#savepath = datadir
savepath = '/data/oceanparcels/output_data/data_Miriam/'
sio.savemat(savepath+'propertiesIrminger_coords.mat', coords)
sio.savemat(savepath+'propertiesIrminger_mean.mat', properties_mean)
sio.savemat(savepath+'propertiesIrminger_anom.mat', properties_anom)
sio.savemat(savepath+'propertiesIrminger_mean_month.mat', properties_mean_month)
sio.savemat(savepath+'propertiesIrminger_anom_month.mat', properties_anom_month)

