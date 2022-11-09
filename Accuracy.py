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
J = 8

# read AK results
infile = open('OLGAK_J'+str(J)+'.pkl', 'rb')
Paths, TPIdata = pkl.load(infile)
infile.close()

# unpack Paths
[Bigc, Bigl, Bigk, Kpath_AK, Lpath, Ypath, rpath, wpath, Fpath, Ipath] = Paths

# read AMF results
infile = open('OLGAMF_J'+str(J)+'.pkl', 'rb')
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
infile = open('OLGSRA_J'+str(J)+'.pkl', 'rb')
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

if J==1:
    # read LAA results
    infile = open('OLGLA_J'+str(J)+'.pkl', 'rb')
    Sim, coeffs, elapsed = pkl.load(infile)
    infile.close()
    
    # unpack Paths
    [Kpath_LA, Lpath, Ypath, Cpath, wpath, rpath, zApath, Fpath, Ihist] = Sim
    
    # calulate distance measures
    MAD = np.mean(np.abs(Kpath_LA/Kpath_AK-1))
    RMSD = np.mean((Kpath_LA/Kpath_AK-1)**2) ** .5
    
    print('LA vs AK')
    print('MAPD:  ', MAD)
    print('RMSPD: ', RMSD)

fig = plt.figure()
plt.plot(Kpath_AK, 'k-', label='AK')
plt.plot(Kpath_AMF, 'r-', label='AMF')
plt.plot(Kpath_SRA, 'b-', label='SRA')
if J==1:
    plt.plot(Kpath_LA, 'g-', label='LA')
plt.legend()
fig.savefig('AccuracyFig_J'+str(J)+'.png', dpi=600)