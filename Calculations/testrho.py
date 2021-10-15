# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:03:41 2021

@author: miria
"""
import scipy.io as sio
import numpy as np
import pandas as pd
import gsw
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean.cm as cmo
import datetime
from datetime import timedelta
from scipy import interpolate
import seaborn as sns
sns.set_style('dark')
from info import datadir

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
rho, alpha, beta = gsw.rho_alpha_beta(SA,CT,p)

S0 = np.nanmean(SA,axis=0)
T0 = np.nanmean(CT,axis=0)
rho0 = gsw.rho(S0,T0,p)
#%%
rho_test = np.zeros(np.shape(rho))
for i in range(len(time)):
    for j in range(len(depth)):
        rho_test[i,j] = rho0[j]*(1-alpha[i,j]*(CT[i,j]-T0[j])+beta[i,j]*(SA[i,j]-S0[j]))
        
rho_diff = rho-rho_test