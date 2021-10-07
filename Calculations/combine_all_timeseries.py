"""
Combining all the relevant timeseries into Pandas DataFrames
"""
import scipy.io as sio
import numpy as np
import datetime
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from info import datadir

#%%
# =============================================================================
# SURFACE FORCING
# =============================================================================


# =============================================================================
# Surface freshwater and heat fluxes
# =============================================================================
data = sio.loadmat(datadir+'surface_fluxes_DCA.mat')

time = data['time'][0]
time = time.astype(np.float)
t0 = datetime.datetime(1900,1,1,0,0)
dates = np.array([t0 + datetime.timedelta(hours=i) for i in time])

E = data['E'][0]
P = data['P'][0]
E_P = data['E_P'][0]
Q = data['Q'][0]


# =============================================================================
# Surface S and T
# =============================================================================

data = sio.loadmat(datadir+'timedepth_DCA.mat')
surfS = data['SA'][:,0]
surfT = data['CT'][:,0]


#%%
# =============================================================================
# CONVECTION/STRATIFICATION
# =============================================================================

# =============================================================================
# Mixed Layer Depth
# =============================================================================

data = sio.loadmat(datadir+'timedepth_DCA.mat')
MLD = data['mld_mean'][0]


# =============================================================================
# Stratification
# =============================================================================

data = sio.loadmat(datadir+'SI.mat')
SI_all_upper = data['SI_all_upper'][0]
SI_all_lower = data['SI_all_lower'][0]
SI_all_total = data['SI_all_total'][0]

SI_T_upper = data['SI_T_upper'][0]
SI_T_lower = data['SI_T_lower'][0]
SI_T_total = data['SI_T_total'][0]

SI_S_upper = data['SI_S_upper'][0]
SI_S_lower = data['SI_S_lower'][0]
SI_S_total = data['SI_S_total'][0]


#%%
# =============================================================================
# SAVE ALL MONTHLY TIMESERIES
# =============================================================================

df = pd.DataFrame({"dates": dates,
                   "E": E, "P": P, "E_P": E_P, "Q": Q,
                   "surfS": surfS, "surfT": surfT,
                   "MLD": MLD,
                   "SI_all_upper": SI_all_upper, "SI_all_lower": SI_all_lower, "SI_all_total": SI_all_total,
                   "SI_S_upper": SI_S_upper, "SI_S_lower": SI_S_lower, "SI_S_total": SI_S_total,
                   "SI_T_upper": SI_T_upper, "SI_T_lower": SI_T_lower, "SI_T_total": SI_T_total
                    })
df.to_csv(datadir+'monthly_timeseries.csv')












#%%
# =============================================================================
# ANNUAL TIMESERIES
# =============================================================================

#%%
# =============================================================================
# SURFACE FORCING
# =============================================================================

month_numbers = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,\
                'September': 9, 'October': 10, 'November': 11, 'December': 12}

def find_month_start(month,dates):
    no = month_numbers[month]
    ind_month_start = []
    for i in np.arange(1,len(dates)):
        if no==1:
            if dates[i-1].month == 12 and dates[i].month == 1:
                ind_month_start.append(i)
        else:
            if dates[i-1].month == no-1 and dates[i].month == no:
                ind_month_start.append(i)
    return ind_month_start


ind_july_start = find_month_start('July',dates)
start_oct = find_month_start('October',dates)
start_apr = find_month_start('April',dates)
days_month = 30 # average number of days in a month


# =============================================================================
# # Accumulated evaporation/precipitation/freshwater gain/heat loss over a year (July-June)
# =============================================================================
E_int = np.zeros(len(ind_july_start))
P_int = np.zeros(len(ind_july_start))
E_P_int = np.zeros(len(ind_july_start))
Q_int = np.zeros(len(ind_july_start))
for i in np.arange(1,len(E_P_int)):
    for j in range(ind_july_start[i-1],ind_july_start[i]):
        E_int[i] += E[j]
        P_int[i] += P[j]
        E_P_int[i] += E_P[j]
        Q_int[i] += Q[j]*days_month

E_int[0] = np.nan
P_int[0] = np.nan
E_P_int[0] = np.nan
Q_int[0] = np.nan


# =============================================================================
# # Accumulated values over the winter (October-March)
# =============================================================================

start_oct_winter = start_oct[0:-1]
start_apr_winter = start_apr[1:]
E_winter = np.zeros(len(start_oct))
P_winter = np.zeros(len(start_oct))
E_P_winter = np.zeros(len(start_oct))
Q_winter = np.zeros(len(start_oct))
for i in np.arange(0,len(Q_winter)-1):
    for j in range(start_oct_winter[i],start_apr_winter[i]):
        E_winter[i+1] += E[j]
        P_winter[i+1] += P[j]
        E_P_winter[i+1] += E_P[j]
        Q_winter[i+1] += Q[j]*days_month
E_winter[0] = np.nan
P_winter[0] = np.nan
E_P_winter[0] = np.nan
Q_winter[0] = np.nan

# =============================================================================
# # Accumulated values over the summer (April-September)
# =============================================================================
E_summer = np.zeros(len(start_oct))
P_summer = np.zeros(len(start_oct))
E_P_summer = np.zeros(len(start_oct))
Q_summer = np.zeros(len(start_oct))
for i in range(len(Q_summer)):
    for j in range(start_apr[i],start_oct[i]):
        E_summer[i] += E[j]
        P_summer[i] += P[j]
        E_P_summer[i] += E_P[j]
        Q_summer[i] += Q[j]*days_month



#%%
# =============================================================================
# CONVECTION/RESTRATIFICATION
# =============================================================================

# Find the maxima/minima in stratification and convection
def findMaxsMins(data,height_maxs=None,distance_maxs=None,height_mins=None,distance_mins=None,plot=False):
    maxs = signal.find_peaks(data,height=height_maxs,distance=distance_maxs)[0]
    mins = signal.find_peaks(data*-1,height=height_mins,distance=distance_mins)[0]
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(12,4))
        ax.plot(time,data)
        ax.set_xlim(time[0],time[-1])
        ax.set_xticks(time[0::24])
        ax.set_xticklabels([dates[i].year for i in np.arange(0,len(time[0::24])*24,24)],fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.scatter(time[maxs],data[maxs],color='black')
        ax.scatter(time[mins],data[mins],color='black')
    return maxs,mins
    
# ALL
maxs_all_total,mins_all_total = findMaxsMins(SI_all_total,distance_maxs=7,distance_mins=8)#,plot=True)
maxs_all_upper,mins_all_upper = findMaxsMins(SI_all_upper,distance_maxs=8,distance_mins=8)#,plot=True)
mins_all_upper[0] = 0 # correct for first min in upper layer
maxs_all_lower,mins_all_lower = findMaxsMins(SI_all_lower,distance_maxs=7,distance_mins=8)#,plot=True)

# ONLY S
maxs_S_total,mins_S_total = findMaxsMins(SI_S_total,distance_maxs=7,distance_mins=8,height_maxs=-0.03)#,plot=True)
maxs_S_upper,mins_S_upper = findMaxsMins(SI_S_upper,distance_maxs=8,distance_mins=8)#,plot=True)
maxs_S_lower,mins_S_lower = findMaxsMins(SI_S_lower,distance_maxs=7,distance_mins=8)#,plot=True)
maxs_S_lower[-1] = -1 # correct for last max in upper layer

# ONLY T
maxs_T_total,mins_T_total = findMaxsMins(SI_T_total,distance_maxs=7,distance_mins=8)#,plot=True)
maxs_T_upper,mins_T_upper = findMaxsMins(SI_T_upper,distance_maxs=7,distance_mins=8)#,plot=True)
maxs_T_lower,mins_T_lower = findMaxsMins(SI_T_lower,distance_maxs=7,distance_mins=8)#,plot=True)

# Finally, also maxs in MLD
maxs_mld = np.hstack(([0],signal.find_peaks(MLD,height=100,distance=3)[0]))

#%%
# =============================================================================
# Compute annual restratification and convection
# =============================================================================

def convection(conv,maxs):
    return np.array([conv[maxs[i]] for i in range(len(maxs))])

def restratification(strat,maxs,mins):
    return np.array([strat[maxs[i]] - strat[mins[i]] for i in range(len(maxs))])


conv = convection(MLD,maxs_mld)

restrat_all_upper = restratification(SI_all_upper,maxs_all_upper,mins_all_upper)
restrat_all_lower = restratification(SI_all_lower,maxs_all_lower,mins_all_lower)
restrat_all_total = restratification(SI_all_total,maxs_all_total,mins_all_total)

restrat_S_upper = restratification(SI_S_upper,maxs_S_upper,mins_S_upper)
restrat_S_lower = restratification(SI_S_lower,maxs_S_lower,mins_S_lower)
restrat_S_total = restratification(SI_S_total,maxs_S_total,mins_S_total)

restrat_T_upper = restratification(SI_T_upper,maxs_T_upper,mins_T_upper)
restrat_T_lower = restratification(SI_T_lower,maxs_T_lower,mins_T_lower)
restrat_T_total = restratification(SI_T_total,maxs_T_total,mins_T_total)



#%%
# =============================================================================
# SAVE ALL ANNUAL TIMESERIES
# =============================================================================

df_annual = pd.DataFrame({"years_start": dates[0::12], "years_half": dates[6::12],
                   "E_int": E_int, "P_int": P_int, "E_P_int": E_P_int, "Q_int": Q_int,
                   "E_winter": E_winter, "P_winter": P_winter, "E_P_winter": E_P_winter, "Q_winter": Q_winter,
                   "E_summer": E_summer, "P_summer": P_summer, "E_P_summer": E_P_summer, "Q_summer": Q_summer,
                   "conv": conv, "maxs_mld": maxs_mld,
                   "restrat_all_upper": restrat_all_upper, "restrat_all_lower": restrat_all_lower, "restrat_all_total": restrat_all_total,
                   "restrat_S_upper": restrat_S_upper, "restrat_S_lower": restrat_S_lower, "restrat_S_total": restrat_S_total,
                   "restrat_T_upper": restrat_T_upper, "restrat_T_lower": restrat_T_lower, "restrat_T_total": restrat_T_total,
                   "mins_all_upper": mins_all_upper, "mins_all_lower": mins_all_lower, "mins_all_total": mins_all_total,
                   "maxs_all_upper": maxs_all_upper, "maxs_all_lower": maxs_all_lower, "maxs_all_total": maxs_all_total
                    })
df_annual.to_csv(datadir+'annual_timeseries.csv')


