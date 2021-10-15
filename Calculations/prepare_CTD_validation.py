"""
Preparing data from ship CTD
"""

import numpy as np
import scipy.io as sio
import gsw
import netCDF4 as nc
import datetime
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cmocean.cm as cmo
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
sns.set_style('dark')
from info import datadir, mooring_west, mooring_east#, axSettingsLC

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




#%%
# =============================================================================
# 2005
# =============================================================================


#%% LOAD DATA

ctd = sio.loadmat(datadir+'CTD2005_raw.mat')
lon = ctd['lon'][0]*-1
lat = ctd['lat'][0]
matrices = ctd['matrices']
P = matrices['P'][0][0]
S = matrices['S'][0][0]
T = matrices['T'][0][0]

#%% COMPARE WITH OSNAP LINE

# Hydrographic section coordinates
data = sio.loadmat(datadir+'hydrographic_sections.mat')
lon_section = data['lon'][0]
lat_section = data['lat'][0]
dist = data['distance'][0]

osnap_ind = np.where((lon>mooring_west[0])&(lon<mooring_east[0]))[0]

#%% Figure to compare
bounds_lon = [-45,-27]
bounds_lat = [58,63]

projection = ccrs.LambertConformal(central_longitude=np.mean(bounds_lon),central_latitude=np.mean(bounds_lat))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,7), subplot_kw={'projection': projection})
axSettingsLC(ax,bounds_lon,bounds_lat)

# Section line
ax.scatter(lon_section,lat_section,transform=ccrs.PlateCarree(),color='k',s=5,zorder=5)

# CTD locations
plt.scatter(lon[osnap_ind],lat[osnap_ind],transform=ccrs.PlateCarree(),color='red')
plt.scatter(lon[4],lat[4],transform=ccrs.PlateCarree(),color='blue') # too far from OSNAP line


#%% SELECT CORRECT DATA

# Manually remove points at the beginning (so that we are only moving eastward in time)
#osnap_points = np.hstack((osnap_ind[0:2],osnap_ind[3:]))
#osnap_points = osnap_ind
osnap_points = osnap_ind[4:]

# Select correct lon, lat, P, S, T data
lon_osnap = lon[osnap_points]
lat_osnap = lat[osnap_points]
P_osnap = P[:,osnap_points]
S_osnap = S[:,osnap_points]
T_osnap = T[:,osnap_points]

#%% COMPUTE DISTANCES
dist_osnap = np.zeros(len(osnap_points))
dist_osnap_mooring = np.zeros(len(osnap_points))
west = np.argsort(lon_osnap)[0] # westernmost point
for i in range(len(osnap_points)):
    dist_osnap[i] = gsw.distance([lon_osnap[i],lon_osnap[west]],[lat_osnap[i],lat_osnap[west]])
    dist_osnap_mooring[i] = gsw.distance([lon_osnap[i],mooring_west[0]],[lat_osnap[i],mooring_west[1]])
    
# Sort by distance
dist_osnap_sortind = np.argsort(dist_osnap)
dist_osnap_sorted = dist_osnap[dist_osnap_sortind]
lon_osnap_sorted = lon_osnap[dist_osnap_sortind]
lat_osnap_sorted = lat_osnap[dist_osnap_sortind]
P_osnap_sorted = P_osnap[:,dist_osnap_sortind]
S_osnap_sorted = S_osnap[:,dist_osnap_sortind]
T_osnap_sorted = T_osnap[:,dist_osnap_sortind]


#%% COMPUTE DEPTH
p = np.arange(1,3501,1)
depth = gsw.z_from_p(p,np.mean(lat_osnap))*-1
# Dist,Depth = np.meshgrid(dist_osnap_sorted,depth)
# plt.figure()
# plt.contourf(Dist,Depth,T_osnap_sorted,cmap='cmo.thermal',levels=15)


#%% COMPUTE SA, CT, SIGMA0
SA = np.zeros(np.shape(S_osnap_sorted))
CT = np.zeros(np.shape(S_osnap_sorted))
sigma0 = np.zeros(np.shape(S_osnap_sorted))
for i in range(len(p)):
    for j in range(len(dist_osnap_sorted)):
        SA[i,j] = gsw.SA_from_SP(S_osnap_sorted[i,j],p[i],lon_osnap_sorted[j],lat_osnap_sorted[j])
        CT[i,j] = gsw.CT_from_pt(SA[i,j],T_osnap_sorted[i,j])
        sigma0[i,j] = gsw.sigma0(SA[i,j],CT[i,j])
        # SA[i,j] = gsw.SA_from_SP(S_osnap[i,j],p[i],lon_osnap[j],lat_osnap[j])
        # CT[i,j] = gsw.CT_from_pt(SA[i,j],T_osnap[i,j])
        # sigma0[i,j] = gsw.sigma0(SA[i,j],CT[i,j])


#%% SAVE DATA
data = {"lon": lon_osnap_sorted, "lat": lat_osnap_sorted, "dist": dist_osnap_sorted,
        "depth": depth, "SA": SA, "CT": CT, "sigma0": sigma0}
sio.savemat(datadir+"CTD2005.mat",data)



# #%%
# Dist,Depth = np.meshgrid(dist_osnap,depth)
# plt.figure()
# plt.contourf(Dist,Depth,T_osnap,cmap='cmo.thermal',levels=15)



#%%
# =============================================================================
# 2016
# =============================================================================

#%% LOAD DATA

ctd = sio.loadmat(datadir+'CTD2016_raw.mat')['CTD']
lon = ctd['lon'][0][0][0]
lat = ctd['lat'][0][0][0]
P = ctd['pres'][0][0]
T = ctd['T'][0][0]
S = ctd['S'][0][0]

#%% COMPARE WITH OSNAP LINE

# Hydrographic section coordinates
data = sio.loadmat(datadir+'hydrographic_sections.mat')
lon_section = data['lon'][0]
lat_section = data['lat'][0]
dist = data['distance'][0]

osnap_ind = np.where((lon>mooring_west[0])&(lon<mooring_east[0]))[0]

#%% Figure to compare
bounds_lon = [-45,-27]
bounds_lat = [58,63]

projection = ccrs.LambertConformal(central_longitude=np.mean(bounds_lon),central_latitude=np.mean(bounds_lat))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,7), subplot_kw={'projection': projection})
axSettingsLC(ax,bounds_lon,bounds_lat)

# Section line
ax.scatter(lon_section,lat_section,transform=ccrs.PlateCarree(),color='k',s=5,zorder=5)

# CTD locations
plt.scatter(lon[osnap_ind],lat[osnap_ind],transform=ccrs.PlateCarree(),color='red')


#%% SELECT CORRECT DATA

osnap_points = osnap_ind

# Select correct lon, lat, P, S, T data
lon_osnap = lon[osnap_points]
lat_osnap = lat[osnap_points]
P_osnap = P[:,osnap_points]
S_osnap = S[:,osnap_points]
T_osnap = T[:,osnap_points]


#%% COMPUTE DISTANCES
dist_osnap = np.zeros(len(osnap_points))
dist_osnap_mooring = np.zeros(len(osnap_points))
for i in range(len(osnap_points)):
    dist_osnap[i] = gsw.distance([lon_osnap[i],lon_osnap[0]],[lat_osnap[i],lat_osnap[0]])
    dist_osnap_mooring[i] = gsw.distance([lon_osnap[i],mooring_west[0]],[lat_osnap[i],mooring_west[1]])
    
# Sort by distance
dist_osnap_sortind = np.argsort(dist_osnap)
dist_osnap_sorted = dist_osnap[dist_osnap_sortind]
lon_osnap_sorted = lon_osnap[dist_osnap_sortind]
lat_osnap_sorted = lat_osnap[dist_osnap_sortind]
P_osnap_sorted = P_osnap[:,dist_osnap_sortind]
S_osnap_sorted = S_osnap[:,dist_osnap_sortind]
T_osnap_sorted = T_osnap[:,dist_osnap_sortind]

#%% COMPUTE DEPTH
p = np.arange(1,1591*2,2)
depth = gsw.z_from_p(p,np.mean(lat_osnap))*-1

#%% COMPUTE SA, CT, SIGMA0
SA = np.zeros(np.shape(S_osnap_sorted))
CT = np.zeros(np.shape(S_osnap_sorted))
sigma0 = np.zeros(np.shape(S_osnap_sorted))
for i in range(len(p)):
    for j in range(len(dist_osnap_sorted)):
        SA[i,j] = gsw.SA_from_SP(S_osnap_sorted[i,j],p[i],lon_osnap_sorted[j],lat_osnap_sorted[j])
        CT[i,j] = gsw.CT_from_pt(SA[i,j],T_osnap_sorted[i,j])
        sigma0[i,j] = gsw.sigma0(SA[i,j],CT[i,j])


#%% SAVE DATA
data = {"lon": lon_osnap_sorted, "lat": lat_osnap_sorted, "dist": dist_osnap_sorted,
        "depth": depth, "SA": SA, "CT": CT, "sigma0": sigma0}
sio.savemat(datadir+"CTD2016.mat",data)








#%% CMEMS DATA
# =============================================================================
# CMEMS data 2005
# =============================================================================


sectiondata = sio.loadmat(datadir+'hydrographic_sections.mat')
lat_section = sectiondata['lat'][0]
lon_section = sectiondata['lon'][0]

#%%
loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'

# Select years + months

# September 2005
years1 ='2003-2007'
t1 = 32
# August 2016
years2 = '2013-2017'
t2 = 43


def makeSectionMonth(years,t):
    fn = loadpath + 'global-reanalysis-phy-001-030-monthly_'+years+'.nc'
    ds = nc.Dataset(fn)
    
    #time = ds['time'][:]
    #t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
    #dates = np.array([t0 + datetime.timedelta(hours=i) for i in time])
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
    
     # Load potential temperature and practical salinity
    PT = ds['thetao'][t,:,lat_min:lat_max+1,lon_min:lon_max+1] # potential temperature
    SP = ds['so'][t,:,lat_min:lat_max+1,lon_min:lon_max+1] # practical salinity
    
    # Interpolate fields to points along section
    SA_s = np.ma.zeros((len(depth),len(lon_section))) # absolute salinity in g/kg
    CT_s = np.ma.zeros((len(depth),len(lon_section))) # conservative temperature in degC
    sigma0_s = np.ma.zeros((len(depth),len(lon_section))) # potential density in kg/m3
    for j in range(len(depth)):            
        # Interpolate fields to points on section
        interp_SP = interpolate.interp2d(lon,lat,SP[j,:,:],kind='linear',fill_value='extrapolate')
        interp_PT = interpolate.interp2d(lon,lat,PT[j,:,:],kind='linear',fill_value='extrapolate')
    
        for k in range(len(lon_section)):
            SP_s = interp_SP(lon_section[k],lat_section[k])
            PT_s = interp_PT(lon_section[k],lat_section[k])
            # From SP and PT, compute SA, CT and sigma0
            p = gsw.p_from_z(depth[j]*-1, lat_section[k])
            SA_s[j,k] = gsw.SA_from_SP(SP_s,p,lat_section[k],lon_section[k])
            CT_s[j,k] = gsw.CT_from_pt(SA_s[j,k],PT_s)
            sigma0_s[j,k] = gsw.sigma0(SA_s[j,k],CT_s[j,k])
    
    SA_s = np.ma.filled(SA_s,np.nan)
    SA_s[SA_s<0]=np.nan
    CT_s = np.ma.filled(CT_s,np.nan)
    sigma0_s = np.ma.filled(sigma0_s,np.nan)
    
    return SA_s, CT_s, sigma0_s


SA_CMEMS_05, CT_CMEMS_05, sigma0_CMEMS_05 = makeSectionMonth(years1,t1)
SA_CMEMS_16, CT_CMEMS_16, sigma0_CMEMS_16 = makeSectionMonth(years2,t2)

data1 = {"SA": SA_CMEMS_05, "CT": CT_CMEMS_05, "sigma0": sigma0_CMEMS_05}
sio.savemat(datadir+'hydrographic_section_2005_sep.mat',data1)

data2 = {"SA": SA_CMEMS_16, "CT": CT_CMEMS_16, "sigma0": sigma0_CMEMS_16}
sio.savemat(datadir+'hydrographic_section_2016_aug.mat',data2)



