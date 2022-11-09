# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:52:20 2022

@author: Kerk
"""

import numpy as np
import pickle as pkl
from chainback import allHHplans
import time
import matplotlib.pyplot as plt


##  SETUP
# set number pf ability types
J = 8

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

# set initital state
Bigk0 = .5*kbarmat          ####
K0 = np.sum(Bigk0)
L0 = .75*Lbar               ####

# start timer
tic = time.perf_counter()
##  MAKE INITIAL GUESS FOR K AND L TIMEPATHS
InitType = 'SS'
if InitType == 'SS':
    # initialize paths to SS values
    Kpath = np.ones(T)*Kbar
    Lpath = np.ones(T)*Lbar
elif InitType == 'Lin':
    # initialize paths using linear trend
    Kpath = np.zeros(T)
    Lpath = np.zeros(T)
    # 1st period
    Kpath[0] = K0
    Lpath[0] = L0
    # up to SS
    for t in range(1,T):
        Kpath[t] = Kpath[t-1] + (Kbar - Kpath[t-1])/(T - t)
        Lpath[t] = Lpath[t-1] + (Lbar - Lpath[t-1])/(T - t)
else:
    InitType = 'Asm'
    # initialize paths using assymptotic approach
    Kpath = np.zeros(T)
    Lpath = np.zeros(T)
    # 1st period
    Kpath[0] = K0
    Lpath[0] = L0
    # up to SS
    for t in range(1,T):
        Kpath[t] = Kpath[0] + (Kbar - Kpath[0]) * t / (t+1)
        Lpath[t] = Lpath[0] + (Lbar - Lpath[0]) * t / (t+1)
   

##  BEGIN TIME PATH ITERATION
maxit = 100         # maximum number of iterations allows
ccrit = 1.0E-6      # convergence criteria
downscale = .5      # how to change rho if dist rises
upscale = .95       # how to change rho if dist falls
dist = 10000.       # initialize distance measure
olddist = dist
itnum = 0           # initialize iteration number
rho = .2            # initialize damping parameter
Kpathiters = Kpath  # initialize record of Kpath over iterations
Lpathiters = Lpath  # initialize record of Lpath over iterations

while(dist>ccrit and itnum<maxit):
    itnum = itnum + 1
    # find values of r, w & f
    Ypath = Kpath**alpha * Lpath**(1-alpha)
    rpath = alpha*Ypath/Kpath
    wpath = (1-alpha)*Ypath/Lpath
    fpath = tau*(wpath*Lpath+(rpath-delta)*Kpath)/(S*J)
    
    # create paths and SSvecs
    paths = [rpath, wpath, fpath]
    SSvecs = [cbarmat, ellbarmat, kbarmat] 
    
    # Solve for all household plans
    Bigc, Bigl, Bigk, Bigcheck = allHHplans(paths, SSvecs, Bigk0, T, params)
            
    ##  AGGREGATE AND CHECK FOR CONVERGENCE
    Kpath2 = np.sum(Bigk, axis = (0,1))
    Lpath2 = np.sum(Bigl*amat3, axis = (0,1))
    Kdist = np.max(np.abs(Kpath-Kpath2))
    Ldist = np.max(np.abs(Lpath-Lpath2)) 
    dist = max(Kdist,Ldist)
    print('itnum: ',itnum,'  dist: ', dist,'  rho: ', rho,'  check max:',\
          np.max(Bigcheck))
    
    # adjust dampener
    if dist > olddist:
        rho = downscale*rho
    else:
        rho = 1 - (1-rho)*upscale
    
    # take convex combination
    Kpath = rho*Kpath2 + (1-rho)*Kpath
    Lpath = rho*Lpath2 + (1-rho)*Lpath
    Kpathiters = np.vstack((Kpathiters, Kpath))
    Lpathiters = np.vstack((Lpathiters, Lpath))
    olddist = dist
    
# end timer
toc = time.perf_counter()
elapsed = toc - tic
print('time elapsed: ', elapsed)
    
##  STORE AND ANALYZE RESULTS

# calculate I and C paths
Ipath = np.zeros(T)
for t in range(0,T-1):
    Ipath[t] = Kpath[t+1] - delta*Kpath[t]
Ipath[T-1] = Kbar - delta*Kpath[T-1]    
Cpath = np.sum(Bigc, axis = (0,1))

# get remainder of paths
Ypath = Kpath**alpha * Lpath**(1-alpha)
rpath = alpha*Ypath/Kpath
wpath = (1-alpha)*Ypath/Lpath
Fpath = tau*(wpath*Lpath - (rpath-delta)*Kpath)
fpath = Fpath/(S*J)

# create lists to save
Paths = [Bigc, Bigl, Bigk, Kpath, Lpath, Ypath, rpath, wpath, Fpath, Ipath]
TPIdata = [itnum, dist, elapsed, Kpathiters, Lpathiters, np.max(Bigcheck), \
    ccrit, maxit, downscale, upscale, InitType]

# save AK information
output = open('OLGAK_J'+str(J)+'.pkl', 'wb')
pkl.dump((Paths, TPIdata), output)
output.close()

# plot transition paths
fig = plt.figure()

plt.subplot(3, 2, 1)
plt.plot(Cpath)
plt.xticks([])
plt.title('Consumption', y=.97)

plt.subplot(3, 2, 3)
plt.plot(Kpath)
plt.xticks([])
plt.title('Capital', y=.97)

plt.subplot(3, 2, 5)
plt.plot(Lpath)
plt.title('Labor', y=.97)

plt.subplot(3, 2, 2)
plt.plot(Ypath)
plt.xticks([])
plt.title('GDP', y=.97)

plt.subplot(3, 2, 4)
plt.plot(rpath)
plt.xticks([])
plt.title('Rental Rate', y=.97)

plt.subplot(3, 2, 6)
plt.plot(wpath)
plt.title('Wage', y=.97)
    
fig.savefig('OLGAKfig_J'+str(J)+'.png', dpi=600)


# plot convergence
fig2 = plt.figure()
plt.plot(Kpathiters[:,0:T].T)
fig2.savefig('OLGAKfig2_J'+str(J)+'.png', dpi=600)


# # plot surface plots
# Saxis = np.linspace(0,S-1,S)
# Taxis = np.linspace(0,T-1,T)
# Sgrid, Tgrid = np.meshgrid(Saxis,Taxis)

# for j in range(0,J):
#     fig3 = plt.figure()
#     # plot surface plots
#     ax = plt.axes(projection ='3d')
#     ax.plot_surface(Sgrid,Tgrid,Bigk[:,j,:].T)
#     plt.title('type '+ str(j))
#     fig3.savefig('OLGAKfig3_J'+str(J)+'_'+str(j)+'.png', dpi=600)


# fig4 = plt.figure()
# for j in range(0,J):
#     plt.plot(np.sum(Bigk[:,j,:], axis=0), label='j='+str(j))
# plt.legend()
# fig2.savefig('OLGAKfig4_J'+str(J)+'.png', dpi=600)