"""
We load the potential temperature, practical salinity and MLD fields in the area
enclosed by the MLD = 650 m contour (averaged over large MLD years), for 1993-2019.
From these we compute the conservative temperature, absolute salinity,
potential density referenced to the surface, buoyancy frequency and 
isopycnal potential vorticity.
We average CT, SA, sigma0, N2, PV and MLD over the (lat-lon) area, so that as a result
we have time series of MLD and of depth profiles of CT, SA, sigma0, N2 and PV. 
"""
import scipy.io as sio
import netCDF4 as nc
import gsw
import numpy as np
#from info import datadir, print_header


#%%
# =============================================================================
# Load DCA area data
# =============================================================================

#mask = np.load(datadir+'DCA_mask.npy')
mask = np.load('/data/oceanparcels/input_data/CMEMS/IrmingerSea/DCA_mask.npy')

# x = np.load(datadir+'DCA_contour_x.npy')
# y = np.load(datadir+'DCA_contour_y.npy')
x = np.load('/data/oceanparcels/input_data/CMEMS/IrmingerSea/DCA_contour_x.npy')
y = np.load('/data/oceanparcels/input_data/CMEMS/IrmingerSea/DCA_contour_y.npy')
bounds_lat = [np.min(y), np.max(y)]
bounds_lon = [np.min(x), np.max(x)]

#%%
# =============================================================================
# Load layer thickness data
# =============================================================================

#loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'
loadpath = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/'

fn = loadpath + 'GLO-MFC_001_024_coordinates.nc'
ds = nc.Dataset(fn)
    
lat = ds['latitude'][:]
lon = ds['longitude'][:]

# Select area of interest
ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]
lat = lat[ind_lat_lower:ind_lat_upper+1] 
lon = lon[ind_lon_left:ind_lon_right+1]

# Load MLD
e3t = ds['e3t'][:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]
e3t = np.ma.filled(e3t,np.nan)
# Flip lat axis so that it matches the mask orientation
e3t = e3t[:,::-1,:]

# Mask points that are outside the area enclosed by the contour
e3t_masked = np.ma.zeros(np.shape(e3t))
for i in range(np.shape(e3t)[0]):
    e3t_masked[i,:,:] = np.ma.masked_array(e3t[i,:,:],mask)


#%%
# =============================================================================
# Load T, S, sigma0, PV, N2 data
# =============================================================================

def profile_timeseries_in_contour(years,bounds_lat,bounds_lon,contour_mask,incl_allvar=False):
    """
    Load CMEMS potential temperature and practical salinity within a certain region;
    compute conservative temperature, absolute salinity, and potential density
    referenced to the surface, and average them over the area.
    
    Parameters:
        years : year range as given in the filename of the .nc file
        bounds_lat, bounds_lon : bounds of latitude and longitude that we select from the file
        contour_mask : mask for points that are outside of the region enclosed by a contour
        incl_depth_mld : if True, also load MLD and average over the region, and return MLD and depth
    """
    #print_header(years+' START')
    #print(years+' START')
    #fn =  '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/global-reanalysis-phy-001-030-monthly_'+years+'.nc'
    fn = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/global-reanalysis-phy-001-030-monthly_'+years+'.nc'
    ds = nc.Dataset(fn)
    
    time = ds['time'][:]
    depth = ds['depth'][:]
    lat = ds['latitude'][:]
    lon = ds['longitude'][:]

    # Select area of interest
    ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
    ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
    ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
    ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]
    lat = lat[ind_lat_lower:ind_lat_upper+1] 
    lon = lon[ind_lon_left:ind_lon_right+1]
    
    # Load potential temperature and practical salinity
    PT = ds['thetao'][:,:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1] # potential temperature
    SP = ds['so'][:,:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1] # practical salinity
    PT = np.ma.filled(PT,np.nan)
    SP = np.ma.filled(SP,np.nan)

    # Flip lat axis so that it matches the mask orientation
    lat = lat[::-1]
    PT = PT[:,:,::-1,:]
    SP = SP[:,:,::-1,:]
    
    #print(years+' loading variables done')
    
    # Mask points that are outside the area enclosed by the contour
    PT_masked = np.ma.zeros(np.shape(PT))
    SP_masked = np.ma.zeros(np.shape(SP))
    for i in range(np.shape(PT)[0]):
        for j in range(np.shape(PT)[1]):
            PT_masked[i,j,:,:] = np.ma.masked_array(PT[i,j,:,:],contour_mask)
            SP_masked[i,j,:,:] = np.ma.masked_array(SP[i,j,:,:],contour_mask)
    #print(years+' masking area done')
    
    # Compute absolute salinity, conservative temperature and potential density
    SA = np.ma.zeros(np.shape(SP)) # absolute salinity 
    CT = np.ma.zeros(np.shape(SP)) # conservative temperature
    sigma0 = np.ma.zeros(np.shape(SP)) # potential density referenced to the surface
    p = np.zeros((len(depth),len(lat))) # sea pressure
    for i in range(len(time)):
        for j in range(len(depth)):
            for k in range(len(lat)):
                p[j,k] = gsw.p_from_z(depth[j]*-1, lat[k])
                for l in range(len(lon)):
                    SA[i,j,k,l] = gsw.SA_from_SP(SP_masked[i,j,k,l], p[j,k], lat[k], lon[l])
                    CT[i,j,k,l] = gsw.CT_from_pt(SA[i,j,k,l], PT_masked[i,j,k,l]) 
                    sigma0[i,j,k,l] = gsw.sigma0(SA[i,j,k,l], CT[i,j,k,l])
    #print(years+' SA, CT, sigma0 done')
    
    # Average CT, SA, sigma0, N2, PV over area (lat+lon), weighted by lat and layer thickness
    weights=np.cos(lat/180*np.pi) # weights for latitude
    CT_area = np.ma.zeros((len(time),len(depth)))
    SA_area = np.ma.zeros((len(time),len(depth)))
    sigma0_area = np.ma.zeros((len(time),len(depth)))
    
    for i in range(len(time)):
        for j in range(len(depth)):
            CT_area[i,j] = np.ma.average(np.ma.average(CT[i,j,:,:],axis=1,weights=e3t[j,:,:]),axis=0,weights=weights)
            SA_area[i,j] = np.ma.average(np.ma.average(SA[i,j,:,:],axis=1,weights=e3t[j,:,:]),axis=0,weights=weights)
            sigma0_area[i,j] = np.ma.average(np.ma.average(sigma0[i,j,:,:],axis=1,weights=e3t[j,:,:]),axis=0,weights=weights)
    CT_area = np.ma.filled(CT_area,np.nan)
    SA_area = np.ma.filled(SA_area,np.nan)
    sigma0_area = np.ma.filled(sigma0_area,np.nan)
    
    if incl_allvar: # also output MLD, time, depth
        #fn2 =  '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/global-reanalysis-phy-001-030-monthly_MLD.nc'
        fn2 = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/global-reanalysis-phy-001-030-monthly_MLD.nc'
        ds2 = nc.Dataset(fn2)
        
        Time = ds2['time'][:]
        mld = ds2['mlotst'][:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]
        mld = np.ma.filled(mld,np.nan)
        mld = mld[:,::-1,:]
        mld_masked = np.ma.zeros(np.shape(mld))
        for i in range(len(Time)):
            mld_masked[i,:,:] = np.ma.masked_array(mld[i,:,:],contour_mask)
        # MLD averaged over area
        mld_area = np.ma.average(np.ma.average(mld_masked,axis=1,weights=weights),axis=1)
        mld_mean = np.ma.filled(mld_area,np.nan)
        # max MLD within area
        mld_max = np.ma.zeros(len(Time))
        for i in range(len(Time)):
            mld_max[i] = np.ma.max(mld_masked[i,:,:])
        return SA_area, CT_area, sigma0_area, mld_mean, mld_max, Time, depth
    else:
        return SA_area, CT_area, sigma0_area



#%%
# =============================================================================
# Compute time-depth profiles
# =============================================================================

decades = ['1993-1996', '1997-2002', '2003-2007', '2008-2012', '2013-2017', '2018-2019']

# Load data per decade
SA1, CT1, sigma01 = profile_timeseries_in_contour(decades[0], bounds_lat, bounds_lon, mask)
SA2, CT2, sigma02 = profile_timeseries_in_contour(decades[1], bounds_lat, bounds_lon, mask)
SA3, CT3, sigma03 = profile_timeseries_in_contour(decades[2], bounds_lat, bounds_lon, mask)
SA4, CT4, sigma04 = profile_timeseries_in_contour(decades[3], bounds_lat, bounds_lon, mask)
SA5, CT5, sigma05 = profile_timeseries_in_contour(decades[4], bounds_lat, bounds_lon, mask)
SA6, CT6, sigma06, mld_mean, mld_max, time, depth = profile_timeseries_in_contour(decades[5], bounds_lat, bounds_lon, mask, True)

# Combine data
SA = np.vstack((SA1,SA2,SA3,SA4,SA5,SA6))
del SA1,SA2,SA3,SA4,SA5,SA6
CT = np.vstack((CT1,CT2,CT3,CT4,CT5,CT6))
del CT1,CT2,CT3,CT4,CT5,CT6
sigma0 = np.vstack((sigma01,sigma02,sigma03,sigma04,sigma05,sigma06))
del sigma01,sigma02,sigma03,sigma04,sigma05,sigma06

# Compute N2 and PV that belong to DCA-averaged properties
N2 = np.zeros((len(time),len(depth)-1))
PV = np.zeros((len(time),len(depth)-1))
lat_mean = np.mean(bounds_lat)
f = 2*7.2921*10**-5*np.sin(lat_mean*np.pi/180) # Coriolis parameter
p = gsw.p_from_z(depth*-1, lat_mean) # pressure values at depths
_, p_mids = gsw.Nsquared(SA[0,:], CT[0,:], p) # pressure midpoints
for i in range(len(time)):
    N2[i,:], _ = gsw.Nsquared(SA[i,:], CT[i,:], p)
    ratio, _ = gsw.IPV_vs_fNsquared_ratio(SA[i,:], CT[i,:], p)
    PV[i,:] = ratio*f*N2[i,:]
    

# Save data in matfile
data = {"time": time, "depth": depth, "p": p, "p_mid": p_mids,
        "SA": SA, "CT": CT, "sigma0": sigma0, "N2": N2, "PV": PV, "mld_mean": mld_mean, "mld_max": mld_max}
#sio.savemat(datadir+"timedepth_DCA.mat", data)
sio.savemat('/data/oceanparcels/output_data/data_Miriam/timedepth_DCA.mat', data)
