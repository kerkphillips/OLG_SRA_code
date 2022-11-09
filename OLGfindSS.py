# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 05:19:38 2022

@author: Kerk
"""

import matplotlib.pyplot as plt
import pickle as pkl
from SSfuncs import getOLGSS, readamat

'''
This program finds the steady state for the OLG model with J ability types

The ability parameters in amat are read from an external Excel file
'''


# set parameters
S = 80
J = 8
beta = .95
gamma = 3.
theta = 2.
chi = 10.
alpha = .35
delta = .08
tau = .2
amat = readamat()
params = [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat]

# set up rwguess
rbarguess = 0.1544
wbarguess = 1.
fbarguess = .1
rwfguess = [rbarguess, wbarguess, fbarguess]

# set initial consumption guess
cbar0guess = .75

OLGseries = getOLGSS(rwfguess, cbar0guess, J, params)
                              
# unpack Series
[cbarmat, ellbarmat, kbarmat, fbarmat, Kbar, Lbar, Cbar, Ybar, \
    Fbar, Ibar, rbar, wbar, fbar] = OLGseries

# plot results
fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(amat[:,0:J])
plt.xticks([])
plt.title('Productivity', y=.97)

plt.subplot(2, 2, 2)
plt.plot(cbarmat)
plt.xticks([])
plt.title('Consumption', y=.97)

plt.subplot(2, 2, 3)
plt.plot(ellbarmat)
plt.title('Labor', y=.97)

plt.subplot(2, 2, 4)
plt.plot(kbarmat[0:-1])
plt.title('Capital', y=.97)
    
fig.savefig('OLGSSfig_J'+str(J)+'.png', dpi=600)

# save SS information
output = open('OLGSS_J'+str(J)+'.pkl', 'wb')
pkl.dump((OLGseries, params), output)
output.close()