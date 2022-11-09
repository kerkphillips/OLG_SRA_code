# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 05:55:22 2022

@author: Kerk
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

##  SETUP
# set number pf ability types
J = 1

# read AK results
infile = open('OLGAK_J'+str(J)+'.pkl', 'rb')
Paths, TPIdata = pkl.load(infile)
infile.close()

# unpack Paths
[Bigc, Bigl, Bigk, Kpath_AK, Lpath, Ypath, rpath, wpath, Fpath, Ipath] = Paths

# read AMF results
infile = open('OLGHybAMF_J'+str(J)+'.pkl', 'rb')
Paths, AMFdata = pkl.load(infile)
infile.close()

# unpack Paths
[Bigc, Bigl, Bigk, Kpath_AMF, Lpath, Ypath, rpath, wpath, Fpath, Ipath] = Paths

# calulate distance measures
MAD = np.mean(np.abs(Kpath_AMF/Kpath_AK-1))
RMSD = np.mean((Kpath_AMF/Kpath_AK-1)**2) ** .5

print('AMF vs AK')
print('MAPD:  ', MAD)
print('RMSPD: ', RMSD)
print('')

# read SRA results
infile = open('OLGHybSRA_J'+str(J)+'.pkl', 'rb')
Paths, TPIdata = pkl.load(infile)
infile.close()

# unpack Paths
[Bigc, Bigl, Bigk, Kpath_SRA, Lpath, Ypath, rpath, wpath, Fpath, Ipath] = Paths

# calulate distance measures
MAD = np.mean(np.abs(Kpath_SRA/Kpath_AK-1))
RMSD = np.mean((Kpath_SRA/Kpath_AK-1)**2) ** .5

print('SRA vs AK')
print('MAPD:  ', MAD)
print('RMSPD: ', RMSD)
print('')

fig = plt.figure()
plt.plot(Kpath_AK, 'k-', label='AK')
plt.plot(Kpath_AMF, 'r-', label='AMF')
plt.plot(Kpath_SRA, 'b-', label='SRA')
plt.legend()
fig.savefig('AccuracyHybFig_J'+str(J)+'.png', dpi=600)