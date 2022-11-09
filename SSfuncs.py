# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 05:19:38 2022

@author: Kerk
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd


def cfuncSS(cbar, rbar, beta, delta, tau, gamma):

    '''
    This function uses the capital Euler equation to find cbar(s+1)
    '''

    cbarp = (beta*(1+(rbar-delta)*(1-tau)))**(1/gamma) * cbar

    return cbarp


def ellfuncSS(cbar, wbar, tau, a, chi, gamma, theta):

    '''
    This function uses the labor-leisure Euler equation to find ellbar(s)
    '''

    ellbar = (((1-tau)*wbar*a)/(chi*cbar**gamma))**(1/theta)

    return ellbar


def kfuncSS(kbar, cbar, ellbar, rbar, wbar, fbar, tau, a, delta):

    '''
    This function used the household budget constraint to find kbar(s+1)
    '''

    kbarp = (1-tau)*(wbar*a*ellbar + (rbar-delta)*kbar) + kbar + fbar - cbar

    return kbarp


def lifecycleSS(c0bar, j, bars, params):
    
    '''
    This function calculates the life-cycle behavior for an agent of age s
    
    inputs are:
        
        c1: initial consumption
        k1: initial capital holdings
        s: initial age
        j: labor productivity type
        vecs: list of numpy vectors which are the histories of interest rates, 
            wages, and transfers over the remaining life of the household
        params; list of model parameters
        
    outputs are:
        
        cvec: numpy vector - history of consumption over the remaining life
        lvec: numpy vector - history of labor over the remaining life
        kvec: numpy vector - history of capital over the remaining life
        
    '''
    
    # unpack vecs
    [rbar, wbar, fbar] = bars
    
    # unpack params
    [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params
    
    # initialize output vectors
    cbarvec = np.empty(S)
    ellbarvec = np.empty(S)
    kbarvec = np.empty(S+1)
    
    # assign initial values
    cbarvec[0] = c0bar
    kbarvec[0] = 0.
    ellbarvec[0] = ellfuncSS(cbarvec[0], wbar, tau, amat[0,j], chi, gamma, 
        theta)
    kbarvec[1] = kfuncSS(kbarvec[0], cbarvec[0], ellbarvec[0], rbar, wbar, \
        fbar, tau, amat[0,j], delta)
    
    # assign values over remaining time periods
    for s in range(1,S):
        cbarvec[s] = cfuncSS(cbarvec[s-1], rbar, beta, delta, tau, gamma)
        ellbarvec[s] = ellfuncSS(cbarvec[s], wbar, tau, amat[s,j], chi, \
            gamma, theta)
        kbarvec[s+1] = kfuncSS(kbarvec[s], cbarvec[s], ellbarvec[s], rbar, \
            wbar, fbar, tau, amat[s,j], delta)
            
    return cbarvec, ellbarvec, kbarvec

    
def findkSp(cbar0, j, bars, params):
    
    '''
    This function returns only the final capital value
    '''
    
    cbarvec, ellbarvec, kbarvec = lifecycleSS(cbar0, j, bars, params)
    
    kbarSp = kbarvec[-1]
    
    return kbarSp


def getOLGseries(inbars, params):
    
    '''
    This function finds the OLG steady state with starting inputs for the 
    initial consumptions by type, interest rate, and wage rate.
    '''
    # unpack params
    [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params
    
    # upack guesses
    c0bar = inbars[0:J]
    rbar = inbars[J]
    wbar = inbars[J+1]
    fbar = inbars[J+2]

    # adjust size of amat
    amat = amat[:,0:J]
    
    # allocate SS arrays
    cbarmat = np.zeros((S,J))
    ellbarmat = np.zeros((S,J))
    kbarmat = np.zeros((S+1,J))
    
    # loop over j to get lifecycle paths for c, ell, and k
    bars = [rbar, wbar, fbar]
    for j in range(0,J):
        cbarmat[:,j], ellbarmat[:,j], kbarmat[:,j] \
            = lifecycleSS(c0bar[j], j, bars, params)
    
    # do aggregations
    Kbar = np.sum(kbarmat)
    Lbar = np.sum(ellbarmat*amat)
    Cbar = np.sum(cbarmat)
    
    # other definitions
    Ybar = Kbar**alpha * Lbar**(1-alpha)
    Fbar = tau*(wbar*Lbar + (rbar-delta)*Kbar)
    Ibar = delta*Kbar
    fbarmat = np.ones((S,J))/(S*J)
    
    # deviations
    rdev = rbar - alpha*Ybar/Kbar
    wdev = wbar - (1-alpha)*Ybar/Lbar
    fdev = fbar - Fbar/(S*J)
    c0dev = np.zeros(J)
    for j in range(0,J):
        c0dev[j] = kbarmat[S,j]
        
    Series = [cbarmat, ellbarmat, kbarmat, fbarmat, Kbar, Lbar, Cbar, Ybar, \
        Fbar, Ibar, rbar, wbar, fbar]
        
    Devs = np.concatenate((c0dev, np.array([rdev, wdev, fdev])))
    
    
    return Series, Devs


def getDevs(inbars, params):
    
    '''
    This function returns only Devs from gerOLGSS
    '''
    
    Series, Devs = getOLGseries(inbars, params)
    
    return Devs


def getOLGSS(rwfguess, cbar0guess, J, params):
    
    '''
    This function calculates the steady state for the OLG model
    
    rwguess is guesses for rbar and wbar
    cbar0guess is a starting guess for initial consumption for all j types
    '''
    
    # iterate over ability types to get better guesses for initial consumption
    
    # allocate vector for gueesses
    cbar0guesses = np.zeros(J)
    for j in range(0,J):
        # optimize
        f = lambda cbar0: findkSp(cbar0, j, rwfguess, params)
        output = opt.root(f, cbar0guess, method='hybr')
        cbar0 = output.x
        check = findkSp(cbar0, j, rwfguess, params)
        print('check1: ', j, '  ', check)
        cbarvec, ellbarvec, kbarvec = lifecycleSS(cbar0, j, rwfguess, params)
        
        # load starting consumption values
        cbar0guesses[j] = cbar0[0]
        
    # now find the steady state
    
    # set up guesses
    [rbarguess, wbarguess, fbarguess] = rwfguess
    guesses = np.concatenate((cbar0guesses, np.array([rbarguess, wbarguess, \
        fbarguess])))

    # optimize
    f = lambda inbars: getDevs(inbars, params)
    output = opt.root(f, guesses, method='hybr')
    inbars = output.x
    check = getDevs(inbars, params)
    print('check2: ', np.max(np.abs(check)))
    print('')
    print('soln:   ', inbars)
    OLGseries, OLGdevs = getOLGseries(inbars, params)
    
    return OLGseries


def readamat():
    
    '''
    This function reads the productivity matrix, amat, from an Excel file.
    
    Ages are in rows and ability types are in columns
    '''
    
    # read data into pandas dataframe
    df = pd.read_excel('amat.xlsx', sheet_name = 'amat', nrows=81, \
        usecols = 'B:I')
        
    # export data to numpy array
    amat = np.asarray(df)
    
    return amat

