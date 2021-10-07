"""
Computing the convection resistance over the two layers in the DCA
"""

import scipy.io as sio
import numpy as np
import gsw
import datetime
import seaborn as sns
sns.set_style('darkgrid')
from info import datadir


#%%
# =============================================================================
# Load layer thickness data
# =============================================================================

# loadpath = '//zeus.nioz.nl/ocs/data/OSNAP/Reanalysis/CMEMS/ocean/monthly/raw/'
# #loadpath = '/data/oceanparcels/input_data/CMEMS/IrmingerSea/'

# fn = loadpath + 'GLO-MFC_001_024_coordinates.nc'
# ds = nc.Dataset(fn)
    
# lat = ds['latitude'][:]
# lon = ds['longitude'][:]

# # Select area of interest
# bounds_lon = [-45,-35]
# bounds_lat = [58,65]
# ind_lat_lower = np.argwhere(lat<=bounds_lat[0])[-1][0]
# ind_lat_upper = np.argwhere(lat>=bounds_lat[1])[0][0]
# ind_lon_left = np.argwhere(lon<=bounds_lon[0])[-1][0]
# ind_lon_right = np.argwhere(lon>=bounds_lon[1])[0][0]
# lat = lat[ind_lat_lower:ind_lat_upper+1] 
# lon = lon[ind_lon_left:ind_lon_right+1]

# # Load layer thickness
# e3t = ds['e3t'][:,ind_lat_lower:ind_lat_upper+1,ind_lon_left:ind_lon_right+1]
# e3t = np.ma.filled(e3t,np.nan)
# e3tmean=np.nanmean(np.nanmean(e3t,axis=1),axis=1)

# data = {"e3t": e3tmean}
# sio.savemat(datadir+'layer_thickness_DCA.mat',data)

e3t = sio.loadmat(datadir+'layer_thickness_DCA.mat')['e3t'][0]

# =============================================================================
# Compute layer bottoms
# =============================================================================

bottoms = np.zeros(len(e3t))
bottoms[0] = e3t[0]
for i in np.arange(1,len(bottoms)):
    bottoms[i] = bottoms[i-1] + e3t[i]
    

#%%
# =============================================================================
# Load S, T, sigma0 data
# =============================================================================

data = sio.loadmat(datadir+'timedepth_DCA.mat')
time = data['time'][0]
t0 = datetime.datetime(1950,1,1,0,0) # origin of time = 1 January 1950, 00:00:00 UTC
dates = np.array([t0 + datetime.timedelta(hours=i) for i in time])
depth = data['depth'][0]
p = data['p'][0]
SA = data['SA']
CT = data['CT']
sigma0 = data['sigma0']


#%%
# =============================================================================
# Divide into two layers
# =============================================================================
layer1_ind = np.where(depth>90)[0][0]
layer2_ind = np.where(depth>500)[0][0]
# Counting the layer bottoms: layer 1 continues until a depth of 100.455 m
# and layer 2 until a depth of 587.409 m.


#%%
# =============================================================================
# Compute convection resistance
# =============================================================================

def computeCR(profile):
    CR_upper = np.zeros(len(time))
    CR_lower = np.zeros(len(time))
    CR_total = np.zeros(len(time))
    
    for i in range(len(time)):
        for j in range(0,layer1_ind+1):
            CR_upper[i] += profile[i,j]*e3t[j]
        CR_upper[i] -= profile[i,layer1_ind]*bottoms[layer1_ind]
        for k in np.arange(layer1_ind+1,layer2_ind+1):
            CR_lower[i] += profile[i,k]*e3t[k]
        CR_lower[i] += profile[i,layer1_ind+1]*bottoms[layer1_ind]
        CR_lower[i] -= profile[i,layer2_ind]*bottoms[layer2_ind]
        for l in range(0,layer2_ind+1):
            CR_total[i] += profile[i,l]*e3t[l]
        CR_total[i] -= profile[i,layer2_ind]*bottoms[layer2_ind]
    
    return CR_upper, CR_lower, CR_total


CR_all_upper, CR_all_lower, CR_all_total = computeCR(sigma0)

#%%
# =============================================================================
# Compute CR contributions of temperature and salinity from linear density approximation
# =============================================================================
T_contrib = np.zeros(np.shape(CT))
S_contrib = np.zeros(np.shape(SA))
for i in range(len(time)):
    alpha = gsw.alpha(SA[i,:],CT[i,:],p)
    beta = gsw.beta(SA[i,:],CT[i,:],p)
    T_contrib[i,:] = CT[i,:]*alpha*-1
    S_contrib[i,:] = SA[i,:]*beta

CR_T_upper, CR_T_lower, CR_T_total = computeCR(T_contrib)
CR_S_upper, CR_S_lower, CR_S_total = computeCR(S_contrib)



#%%
# =============================================================================
# Save data
# =============================================================================

data = {'SI_all_upper': CR_all_upper*-1, 'SI_all_lower': CR_all_lower*-1, 'SI_all_total': CR_all_total*-1,
        'SI_T_upper': CR_T_upper*-1, 'SI_T_lower': CR_T_lower*-1, 'SI_T_total': CR_T_total*-1,
        'SI_S_upper': CR_S_upper*-1, 'SI_S_lower': CR_S_lower*-1, 'SI_S_total': CR_S_total*-1}
sio.savemat(datadir+'SI.mat',data)

