"""
Making hydrographic sections of salinity, temperature, density and eddy kinetic
energy in the Irminger Sea
"""
import scipy.io as sio
import netCDF4 as nc
import gsw
import numpy as np
import datetime
from scipy import interpolate


#%%
# =============================================================================
# MAKE SECTION
# =============================================================================

# Mooring locations
mooring_west = [-42.82545, 60.07013333]
mooring_east = [-30-31.765/60, 58+52.333/60]

# Determine number of points on the section
l_des = 5000 # desired maximum length between points on the section in m
l_tot = gsw.distance([mooring_west[0],mooring_east[0]], [mooring_west[1],mooring_east[1]]) # total distance of section
n_points = int(np.ceil(l_tot/l_des)) # desired number of points along section
l_section = np.linspace(0,l_tot,n_points)[:,0] # desired distances of points along section from western mooring

# Interpolate lon and lat coordinates of western and eastern mooring as a function of distance
interp_lon = interpolate.interp1d([0,l_tot],[mooring_west[0],mooring_east[0]],kind='linear',fill_value='extrapolate')
interp_lat = interpolate.interp1d([0,l_tot],[mooring_west[1],mooring_east[1]],kind='linear',fill_value='extrapolate')

# Find lon and lat coordinates corresponding with desired distances
lon_section = np.array(interp_lon(l_section),dtype='float64')
lat_section = np.array(interp_lat(l_section),dtype='float64')



#%%
# =============================================================================
# LOAD DATA
# =============================================================================

#loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'
loadpath = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/'

def load_section_data(years, lon_section, lat_section, return_depth=False):
    """
    Compute velocities, absolute salinity, conservative temperature and
    potential density along a section in the Irminger Sea.
    
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
    
    # Restrict to area around section
    lat_min = np.where(lat<np.min([mooring_west[1],mooring_east[1]]))[0][-1]
    lat_max = np.where(lat>np.max([mooring_west[1],mooring_east[1]]))[0][0]
    lon_min = np.where(lon<np.min([mooring_west[0],mooring_east[0]]))[0][-1]
    lon_max = np.where(lon>np.max([mooring_west[0],mooring_east[0]]))[0][0]
    lat = lat[lat_min:lat_max+1]
    lon = lon[lon_min:lon_max+1]
    
    # Load velocities
    uo = ds['uo'][:,:,lat_min:lat_max+1,lon_min:lon_max+1] # zonal velocity
    vo = ds['vo'][:,:,lat_min:lat_max+1,lon_min:lon_max+1] # meridional velocity

    # Load potential temperature and practical salinity
    PT = ds['thetao'][:,:,lat_min:lat_max+1,lon_min:lon_max+1] # potential temperature
    SP = ds['so'][:,:,lat_min:lat_max+1,lon_min:lon_max+1] # practical salinity
    
    # Interpolate fields to points along section
    uo_s = np.ma.zeros((len(time),len(depth),len(lon_section))) # longitudinal velocity in m/s
    vo_s = np.ma.zeros((len(time),len(depth),len(lon_section))) # latitudinal velocity in m/s
    SA_s = np.ma.zeros((len(time),len(depth),len(lon_section))) # absolute salinity in g/kg
    CT_s = np.ma.zeros((len(time),len(depth),len(lon_section))) # conservative temperature in degC
    sigma0_s = np.ma.zeros((len(time),len(depth),len(lon_section))) # potential density in kg/m3
    for i in range(len(time)):
        for j in range(len(depth)):            
            # Interpolate fields to points on section
            interp_uo = interpolate.interp2d(lon,lat,uo[i,j,:,:],kind='linear',fill_value='extrapolate')
            interp_vo = interpolate.interp2d(lon,lat,vo[i,j,:,:],kind='linear',fill_value='extrapolate')
            interp_SP = interpolate.interp2d(lon,lat,SP[i,j,:,:],kind='linear',fill_value='extrapolate')
            interp_PT = interpolate.interp2d(lon,lat,PT[i,j,:,:],kind='linear',fill_value='extrapolate')
    
            for k in range(len(lon_section)):
                uo_s[i,j,k] = interp_uo(lon_section[k],lat_section[k])
                vo_s[i,j,k] = interp_vo(lon_section[k],lat_section[k])
                SP_s = interp_SP(lon_section[k],lat_section[k])
                PT_s = interp_PT(lon_section[k],lat_section[k])
                # From SP and PT, compute SA, CT and sigma0
                p = gsw.p_from_z(depth[j]*-1, lat_section[k])
                SA_s[i,j,k] = gsw.SA_from_SP(SP_s,p,lat_section[k],lon_section[k])
                CT_s[i,j,k] = gsw.CT_from_pt(SA_s[i,j,k],PT_s)
                sigma0_s[i,j,k] = gsw.sigma0(SA_s[i,j,k],CT_s[i,j,k])
        #print("Time "+str(i)+" done")
        
    uo_s = np.ma.filled(uo_s,np.nan)
    vo_s = np.ma.filled(vo_s,np.nan)
    SA_s = np.ma.filled(SA_s,np.nan)
    CT_s = np.ma.filled(CT_s,np.nan)
    sigma0_s = np.ma.filled(sigma0_s,np.nan)
    
    if return_depth:    
        return uo_s, vo_s, SA_s, CT_s, sigma0_s, time, depth
    else:
        return uo_s, vo_s, SA_s, CT_s, sigma0_s, time            


# =============================================================================
# Load data for each decade
# =============================================================================


decades = ['1993-1996', '1997-2002', '2003-2007', '2008-2012', '2013-2017', '2018-2019']
uo1, vo1, SA1, CT1, sigma01, time1 = load_section_data(decades[0], lon_section, lat_section)
uo2, vo2, SA2, CT2, sigma02, time2 = load_section_data(decades[1], lon_section, lat_section)
uo3, vo3, SA3, CT3, sigma03, time3 = load_section_data(decades[2], lon_section, lat_section)
uo4, vo4, SA4, CT4, sigma04, time4 = load_section_data(decades[3], lon_section, lat_section)
uo5, vo5, SA5, CT5, sigma05, time5 = load_section_data(decades[4], lon_section, lat_section)
uo6, vo6, SA6, CT6, sigma06, time6, depth = load_section_data(decades[5], lon_section, lat_section, return_depth=True)

# Combine data
time = np.hstack((time1,time2,time3,time4,time5,time6))
del time1,time2,time3,time4,time5,time6
uo = np.vstack((uo1,uo2,uo3,uo4,uo5,uo6))
del uo1,uo2,uo3,uo4,uo5,uo6
vo = np.vstack((vo1,vo2,vo3,vo4,vo5,vo6))
del vo1,vo2,vo3,vo4,vo5,vo6
SA = np.vstack((SA1,SA2,SA3,SA4,SA5,SA6))
del SA1,SA2,SA3,SA4,SA5,SA6
CT = np.vstack((CT1,CT2,CT3,CT4,CT5,CT6))
del CT1,CT2,CT3,CT4,CT5,CT6
sigma0 = np.vstack((sigma01,sigma02,sigma03,sigma04,sigma05,sigma06))
del sigma01,sigma02,sigma03,sigma04,sigma05,sigma06

# Compute EKE
u_mean = np.mean(uo,axis=0)
v_mean = np.mean(vo,axis=0)
u_anom = uo - u_mean
v_anom = vo - v_mean
EKE = (1/2)*((u_anom)**2+(v_anom)**2)



#%%
# =============================================================================
# MEAN OVER ALL TIME
# =============================================================================
S_mean = np.mean(SA,axis=0)
T_mean = np.mean(CT,axis=0)
sigma_mean = np.mean(sigma0,axis=0)
EKE_mean = np.mean(EKE,axis=0)



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
        data_clim[i] = np.mean([data[j,:,:] for j in months[i]],axis=0)
    data_clim_mean = np.vstack([np.expand_dims(data_clim[j],0) for j in range(12)])
    data_clim_anom = np.zeros(np.shape(data))
    for t in range(len(time)):
        m = dates[t].month - 1
        data_clim_anom[t,:,:] = data[t,:,:] - data_clim_mean[m,:,:]
    return data_clim_mean, data_clim_anom


S_clim, S_anom = compMeanAndAnomalyMonth(SA)
T_clim, T_anom = compMeanAndAnomalyMonth(CT)
sigma_clim, sigma_anom = compMeanAndAnomalyMonth(sigma0)
EKE_clim, EKE_anom = compMeanAndAnomalyMonth(EKE)




#%%
# =============================================================================
# Fix NaN values for S and EKE
# =============================================================================

SA[np.where(np.isnan(CT))] = np.nan
S_mean[np.where(np.isnan(T_mean))] = np.nan
S_clim[np.where(np.isnan(T_clim))] = np.nan
S_anom[np.where(np.isnan(T_anom))] = np.nan
EKE[np.where(np.isnan(CT))] = np.nan
EKE_mean[np.where(np.isnan(T_mean))] = np.nan
EKE_clim[np.where(np.isnan(T_clim))] = np.nan
EKE_anom[np.where(np.isnan(T_anom))] = np.nan


#%%
# =============================================================================
# MLD
# =============================================================================

fn = loadpath+'global-reanalysis-phy-001-030-monthly_MLD.nc'
ds = nc.Dataset(fn)
    
time = ds['time'][:]
#depth = ds['depth'][:]
lat = ds['latitude'][:]
lon = ds['longitude'][:]

# Restrict to area around section
lat_min = np.where(lat<np.min([mooring_west[1],mooring_east[1]]))[0][-1]
lat_max = np.where(lat>np.max([mooring_west[1],mooring_east[1]]))[0][0]
lon_min = np.where(lon<np.min([mooring_west[0],mooring_east[0]]))[0][-1]
lon_max = np.where(lon>np.max([mooring_west[0],mooring_east[0]]))[0][0]
lat = lat[lat_min:lat_max+1]
lon = lon[lon_min:lon_max+1]

# Load MLD
mld = ds['mlotst'][:,lat_min:lat_max+1,lon_min:lon_max+1]

# Interpolate fields to points along section
mld_s = np.zeros((len(time),len(lon_section)))
for i in range(len(time)):  
    interp_mld = interpolate.interp2d(lon,lat,mld[i,:,:],kind='linear',fill_value='extrapolate')
    for k in range(len(lon_section)):
        mld_s[i,k] = interp_mld(lon_section[k],lat_section[k])

mld_mean = np.mean(mld_s,axis=0)
mld_clim = np.empty(12,dtype=object)
for i in range(12):
    mld_clim[i] = np.mean([mld_s[j,:] for j in months[i]],axis=0)
mld_clim = np.vstack([np.expand_dims(mld_clim[j],0) for j in range(12)])


#%%
# =============================================================================
# SAVE DATA
# =============================================================================

section = {"time": time, "depth": depth, "distance": l_section, "lat": lat_section, "lon": lon_section,
           "S": SA, "T": CT,"sigma": sigma0, "EKE": EKE, "mld": mld_s,
           "S_mean": S_mean, "T_mean": T_mean, "sigma_mean": sigma_mean, "EKE_mean": EKE_mean, "mld_mean": mld_mean,
           "S_clim": S_clim, "T_clim": T_clim, "sigma_clim": sigma_clim, "EKE_clim": EKE_clim, "mld_clim": mld_clim,
           "S_anom": S_anom, "T_anom": T_anom, "sigma_anom": sigma_anom, "EKE_anom": EKE_anom}

#savepath = datadir
savepath = '/data/oceanparcels/output_data/data_Miriam/'
sio.savemat(savepath+'hydrographic_sections.mat', section)

