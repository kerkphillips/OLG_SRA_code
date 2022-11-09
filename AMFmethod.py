# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:31:55 2022

@author: Kerk
"""

import numpy as np
import pickle as pkl
from chainback import optHHplan
import time
import matplotlib.pyplot as plt

def AMFHHplans(paths, SSvecs, Bigk0, T, params):
    
    '''
    This function solves for the optimal plans of all households over the 
    course of all periods in the simulation.  This is used only for the AMF
    solution method.
    
    It calls the function optHHplan from chainback.py
    
    Inputs are:
        paths:  list of time paths for r, w, and f. Each is a T-period long
            numpy vector
        SSvec:  a list of the steady state lifecycles for c, ell, and k.  Each
            is an S-by-J numpy array.
        Bigk0:  the initial state in period 0, an S-by-J numpy array.
        T:      the number o0f periods in the time path
        params: list of parameter values
        
    Outputs are: 
        Bigc:     an S-by-J-by-T numpy array of household consumptions
        Bigl:     an S-by-J-by-T numpy array of household labor supplies
        Bigk:     an S-by-J-by-T numpy array of household capital holdings
        Bigcheck: a numpy vector of maximum absolute deviations from the 
            solution to the target value for initial capital holdings
    '''
    
    # unpack paths
    [rpath, wpath, fpath] = paths
    
    # unpack SSvecs
    [cbarmat, ellbarmat, kbarmat] = SSvecs
    
    # unpack params
    [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params
    
    # set maximum period to S+1 or T
    T = min(S+1,T)
    
    # arrays to store behavior of all households
    Bigc = np.zeros((S,J,T))
    Bigl = np.zeros((S,J,T))
    Bigk = np.zeros((S,J,T))
    # expanding array of checks
    Bigcheck = np.array([])

    ##  FIND OPTIMAL PLANS FOR ALL HOUSEHOLDS
    # loop over types
    for j in range(0,J):
        # loop over currently living HH
        for s in range(0,S):
            vecs = [rpath[0:S-s], wpath[0:S-s], fpath[0:S-s]]
            cvec, ellvec, kvec, check = \
                optHHplan(cbarmat[S-1,j], 0., S, Bigk0[s,j], s, j, \
                vecs, params)
            Bigcheck = np.append(Bigcheck, check)
            # store results in Big matrices
            for t in range(0,S-s):
                Bigc[s+t,j,t] = cvec[t]
                Bigl[s+t,j,t] = ellvec[t]
                Bigk[s+t,j,t] = kvec[t]

        # loop over those between periods 0 and T
        for t in range(1,T-S):
            vecs = [rpath[t:t+S], wpath[t:t+S], fpath[t:t+S]]
            cvec, ellvec, kvec, check = \
                optHHplan(cbarmat[S-1,j], 0., S, 0., 0, j, \
                vecs, params)
            Bigcheck = np.append(Bigcheck, check)
            # store results in Big matrices
            for s in range(0,S):
                Bigc[s,j,t+s] = cvec[s]
                Bigl[s,j,t+s] = ellvec[s]
                Bigk[s,j,t+s] = kvec[s]
                
                
    return Bigc, Bigl, Bigk, Bigcheck


##  SETUP
# set number pf ability types
J = 1

# get results from SS solution
infile = open('OLGSS_J'+str(J)+'.pkl', 'rb')
OLGseries, params = pkl.load(infile)
infile.close()

# unpack OLGseries
[cbarmat, ellbarmat, kbarmat, fbarmat, Kbar, Lbar, Cbar, Ybar, \
    Fbar, Ibar, rbar, wbar, fbar] = OLGseries
    
# unpack params
[S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params

# set T value (time to SS)
T = 150

# construct productivity matrix over time
amat3 = np.zeros((S,J,T))
for t in range(0,T):
    amat3[:,:,t] = amat[:,0:J]
    
# choose forecasting model
InitType = 'Lin'

# start timer
tic = time.perf_counter()

# initialize forecasts
Kpath = np.zeros(T)
Lpath = np.zeros(T)

# set initital state
Bigk0 = .5*kbarmat[0:S]          ####
K0 = np.sum(Bigk0)
L0 = .75*Lbar               ####

# arrays to store transition paths
TPc = np.zeros((S,J,T))
TPl = np.zeros((S,J,T))
TPk = np.zeros((S,J,T+1))
TPK = np.zeros(T+1)
TPL = np.zeros(T)
TPY = np.zeros(T)
TPr = np.zeros(T)
TPw = np.zeros(T)
TPf = np.zeros(T)
TPC = np.zeros(T)

# set values for capital in period 0
TPk[:,:,0] = Bigk0
TPK[0] = np.sum(Bigk0)
    
##  BEGIN ITERATIONS OVER THE TIMEPATH
for t in range(0,T):

    # forecast K and L
    if InitType == 'Lin':
        # linear trend
        # 1st period
        Kpath[0] = K0
        Lpath[0] = L0
        # up to SS
        for t1 in range(1,T-t):
            Kpath[t1] = Kpath[t1-1] + (Kbar - Kpath[t1-1])/(T - t1)
            Lpath[t1] = Lpath[t1-1] + (Lbar - Lpath[t1-1])/(T - t1)
    else:
        InitType = 'Asm'
        # assymptotic approach
        # 1st period
        Kpath[0] = K0
        Lpath[0] = L0
        # up to SS
        for t in range(1,T-t):
            Kpath[t1] = Kpath[0] + (Kbar - Kpath[0]) * t1 / (t1+1)
            Lpath[t1] = Lpath[0] + (Lbar - Lpath[0]) * t1 / (t1+1)
    
    # derive forecast for other necessary variables
    Ypath = Kpath**alpha * Lpath**(1-alpha)
    rpath = alpha*Ypath/Kpath
    wpath = (1-alpha)*Ypath/Lpath
    fpath = tau*(wpath*Lpath+(rpath-delta)*Kpath)/(S*J)
    
    # create paths and SSvecs
    paths = [rpath, wpath, fpath]
    SSvecs = [cbarmat, ellbarmat, kbarmat] 
    
    # solve for all household plans for next S periods only
    Bigc, Bigl, Bigk, Bigcheck = AMFHHplans(paths, SSvecs, Bigk0, T, params)
    
    # reset K0, L0, Bigk0 to values from 2nd period for next time period
    Bigk0 = Bigk[:,:,1]
    K0 = np.sum(Bigk0)
    L0 = np.sum(Bigl[:,:,0]*amat3[:,:,t])
    
    # store only initial period in transition paths
    TPc[:,:,t] = Bigc[:,:,0]
    TPl[:,:,t] = Bigl[:,:,0]
    TPk[:,:,t+1] = Bigk0
    TPK[t] = Kpath[0]
    TPL[t] = np.sum(Bigl[:,:,0]*amat3[:,:,t])
    TPC[t] = np.sum(Bigc[:,:,0])
    
    print('time period: ', t, '  check max: ', np.max(Bigcheck))
    
# end timer
toc = time.perf_counter()
elapsed = toc - tic
print('time elapsed: ', elapsed)

# calculate I and C paths
TPI = np.zeros(T)
for t in range(0,T-1):
    TPI[t] = TPK[t+1] - delta*TPK[t]
TPI[T-1] = Kbar - delta*TPK[T-1]    

# truncate TPK
TPK = TPK[0:T]

# get remainder of paths
TPY = TPK**alpha * TPL**(1-alpha)
TPr = alpha*TPY/TPK
TPw = (1-alpha)*TPY/TPL
TPF = tau*(TPw*TPL - (TPr-delta)*TPK)

# create lists to save
Paths = [Bigc, Bigl, Bigk, TPK, TPL, TPY, TPr, TPw, TPF, TPI]
AMFdata = [elapsed, np.max(Bigcheck), InitType]

# save AK information
output = open('OLGAMF_J'+str(J)+'.pkl', 'wb')
pkl.dump((Paths, AMFdata), output)
output.close()

# plot transition paths
fig = plt.figure()

plt.subplot(3, 2, 1)
plt.plot(TPC)
plt.xticks([])
plt.title('Consumption', y=.97)

plt.subplot(3, 2, 3)
plt.plot(TPK)
plt.xticks([])
plt.title('Capital', y=.97)

plt.subplot(3, 2, 5)
plt.plot(TPL)
plt.title('Labor', y=.97)

plt.subplot(3, 2, 2)
plt.plot(TPY)
plt.xticks([])
plt.title('GDP', y=.97)

plt.subplot(3, 2, 4)
plt.plot(TPr)
plt.xticks([])
plt.title('Rental Rate', y=.97)

plt.subplot(3, 2, 6)
plt.plot(TPw)
plt.title('Wage', y=.97)
    
fig.savefig('OLGAMFfig_J'+str(J)+'.png', dpi=600)