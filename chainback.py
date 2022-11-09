# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 06:08:56 2022

@author: Kerk
"""

import numpy as np
import scipy.optimize as opt

def cfunc(cp, beta, r, delta, tau, gamma):
    
    c = (beta*(1 + (r-delta)*(1-tau)))**(-1/gamma) * cp
    
    return c


def lfunc(c, a, tau, w, chi, gamma, theta):
    
    l = (((1-tau)*w*a)/(chi*c**gamma))**(1/theta)
    
    return l


def kfunc(kp, c, l, a, tau, w, f, r, delta):
    
    k = (kp-(1-tau)*w*a*l-f+c)/(1+(1-tau)*(r-delta))
    
    return k


def backchain(cf, kf, sf, si, j, rwf, params):
    
    '''
    This function solves backward from a final consumption and capital to
    the initial state, for consumption, labor, and capital
    
    Inputs are:
        cf:     value of final consumption in period sf
        kf:     value of final capital in period sf+1
        sf:     final period
        si:     initial period
        j:      household type
        rwf:    list containing rvec, wvec, and fvec
                which are numpy arrays from si to sp for r and w
        params: list of model parameters
        
    Outputs are:
        cvec:  numpy array of consumption from si to sp
        lvec:  numpy array of labor from si to sp
        kvec:  numpy array of capital from si to +1
    '''
    
    # unpack params
    [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params
    
    # unpack rw
    [rvec, wvec, fvec] = rwf
    
    # find length of time
    # reduce p by one for use in calling elements in arrays
    # but p in assigning vector size is one larger
    p = sf - si - 1
    
    # initialize vectors
    cvec = np.zeros(p+1)
    lvec = np.zeros(p+1)
    kvec = np.zeros(p+2)
    
    # reduce p by one for use in calling elements in arrays
    # assign final values
    kvec[p+1] = kf
    cvec[p] = cf
    lvec[p] = lfunc(cvec[p], amat[sf-1,j], tau, wvec[p], chi, gamma, theta)
    kvec[p] = kfunc(kvec[p+1], cvec[p], lvec[p], amat[sf-1,j], tau, wvec[p], \
        fvec[p], rvec[p], delta)
    
    # solve backward over time to initial period
    for i in range(1,p+1):
        cvec[p-i] = cfunc(cvec[p-i+1], beta, rvec[p-i+1], delta, tau, gamma)
        lvec[p-i] = lfunc(cvec[p-i], amat[sf-1-i,j], tau, wvec[p], chi, gamma, \
            theta)
        kvec[p-i] = kfunc(kvec[p-i+1], cvec[p-i], lvec[p-i], amat[sf-1-i,j], \
            tau, wvec[p-i], fvec[p-i], rvec[p-i], delta)
            
    return cvec, lvec, kvec


def kdist(cf, kf, sf, ktar, si, j, rwf, params):
    
    '''
    This returns the difference between generated ki and a target value
    
    Inputs are:
        cf:     value of final consumption in period sf
        kf:     value of final capital in period sf+1
        sf:     final period
        ktar:   target value for ki
        si:     initial period
        j:      household type
        rwf:    list containing rvec, wvec, and fvec
                which are numpy arrays from si to sp for r and w
        params: list of model parameters
        
    Outputs are:
        cvec:  numpy array of consumption from si to sp
        lvec:  numpy array of labor from si to sp
        kvec:  numpy array of capital from si to +1
    '''    
    
    cvec, lvec, kvec = backchain(cf, kf, sf, si, j, rwf, params)
    
    return kvec[0] - ktar


def optHHplan(cfguess, kf, sf, ktar, si, j, rwf, params):
    
    f = lambda cf: kdist(cf, kf, sf, ktar, si, j, rwf, params)
    output = opt.root(f, cfguess, method='hybr')
    cf = output.x
    check = kdist(cf, kf, sf, ktar, si, j, rwf, params)
    cvec, lvec, kvec = backchain(cf, kf, sf, si, j, rwf, params)
    
    return cvec, lvec, kvec, check


def allHHplans(paths, SSvecs, Bigk0, T, params):
    
    '''
    This function solves for the optimal plans of all households over the 
    course of all periods in the simulation.  This is used for the AK and SRA
    solution methods.
    
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
                
        # loop over those who will be alive in period T
        for t in range(T-S,T):
            vecs = [rpath[t:T], wpath[t:T], fpath[t:T]]
            cvec, ellvec, kvec, check = \
                optHHplan(cbarmat[T-t-1,j], kbarmat[T-t,j], T-t, 0., 0, j, \
                vecs, params)
            Bigcheck = np.append(Bigcheck, check)
            # store results in Big matrices
            for s in range(0,T-t):
                Bigc[s,j,t+s] = cvec[s]
                Bigl[s,j,t+s] = ellvec[s]
                Bigk[s,j,t+s] = kvec[s]
                
    return Bigc, Bigl, Bigk, Bigcheck