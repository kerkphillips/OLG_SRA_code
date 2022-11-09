# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:14:43 2022

@author: Kerk
"""

import numpy as np
import scipy.optimize as opt
from LinApp import LinApp_Deriv, LinApp_Solve
from RAmodel import Modeldyn, generatehist

def SSeqns(inputs, modparams):
    
    '''
    This function returns Euler equation and target deviations given guesses
    for Kbar, beta, chi, and tau
    '''
    
    # unpack inputs
    [Kbar, beta, chi, tau] = inputs
    
    # unpack params
    [S, J, gamma, theta, alpha, delta, amat, CYtar, Ltar, FYtar] = modparams
    
    Lbar = Ltar
    Ybar = Kbar**alpha*Lbar**(1-alpha)
    wbar = (1-alpha)*Ybar/Lbar
    rbar = alpha*Ybar/Kbar
    Fbar = tau*(wbar*Lbar + (rbar-delta)*Kbar)
    Cbar = wbar*Lbar + (rbar-delta)*Kbar
    Ibar = delta*Kbar
    
    KEuler = beta*(1+(1-tau)*(rbar-delta)) - 1
    LEuler = Cbar**(-gamma)*(1-tau)*wbar - chi*Lbar**theta
    CYdev = Cbar/Ybar - CYtar
    FYdev = Fbar/Ybar - FYtar
    
    return Ybar, wbar, rbar, Fbar, Cbar, Ibar, KEuler, LEuler, CYdev, FYdev


def getdevs(inputs, params):
    
    '''
    This function returns only the deviations as a numpy vector
    '''
    
    Ybar, wbar, rbar, Fbar, Cbar, Ibar, KEuler, LEuler, CYdev, FYdev \
        = SSeqns(inputs, params)
        
    return np.array([KEuler, LEuler, CYdev, FYdev])


def recalibrate(guesses, modparams):
    
    '''
    This functions takes guesses for Kbar, beta, chi, and tau and solves for 
    the steady state of the RA model using the parameters and target values
    in the list modparams.
    
    It returns a list of recalibrated parameters, new steady state values,
    and a check that the solution is correct.
    '''
    
    # unpack modparams
    [S, J, gamma, theta, alpha, delta, amat, CYtar, Ltar, FYtar] = modparams
    
    # optimize to find steady state
    f = lambda inputs: getdevs(inputs, modparams)
    fout = opt.root(f, guesses, method='hybr')
    soln = fout.x

    # unpack solution
    [Kbar, beta, chi, tau] = soln
    Lbar = Ltar

    # check solution
    devs = getdevs(soln, modparams)
    check = np.max(np.abs(devs))
    # print('check: ', check)
    if check > 1.0E-8:
        print('Did Not Converge!!')

    # find remaining SS values
    Ybar, wbar, rbar, Fbar, Cbar, Ibar, KEuler, LEuler, CYdev, FYdev \
        = SSeqns(soln, modparams)
        
    # report bars and parameters
    newbars = [Kbar, Lbar, Ybar, wbar, rbar, Fbar, Cbar, Ibar]
    newparams = [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat]
    
    return newparams, newbars, check


def recalpath(targets, guesses, relK0OLG, T, params):
    
    '''
    This function recalibrates the RA model and then runs a new simulation of
    the transition path
    
    Inputs are:
        targets:  a list of 3 targets for C/Y, L, and F/Y
        guesses:  guesses for values ofKbar, beta, chi, and tau
        relK0OLG: the starting value of K relative to its SS value
        T:        the length of the transition path
        params:   parameters for the OLG model
        
    Outputs are:
        Histlist: list of values along the transition path
        RAparams: updated values for RA model parameters
    '''
    
    # unpack targets
    [CYtar, Ltar, FYtar] = targets
    
    # unpack params
    [S, J, beta, gamma, theta, chi, alpha, delta, tau, amat] = params
    
    # set up parameter list for recalibration
    modparams = [S, J, gamma, theta, alpha, delta, amat, CYtar, Ltar, FYtar]

    # recalibrate
    RAparams, RAbars, check = recalibrate(guesses, modparams)

    # unpack
    [KbarRA, LbarRA, YbarRA, wbarRA, rbarRA, FbarRA, CbarRA, IbarRA] = RAbars
    [S, J, betaRA, gammaRA, theta, chiRA, alpha, delta, tau, amat] = RAparams

    # setup bar vectors
    XXbar = np.array([KbarRA])
    YYbar = np.array([LbarRA])
    ZZbar = np.array([0.])
    XXYYbar = np.array([KbarRA, LbarRA])

    # set LinApp parameters
    nXX = 1
    nYY = 1
    nZZ = 1
    logX = 0
    Sylv = 0
    rho_A = .9
    sigma_A = .01

    # write SIMparams with sig_A and si
    SIMparams = [alpha, beta, gamma, delta, chi, theta, tau, rho_A, sigma_A, \
        ZZbar, nXX, nYY, nZZ, logX, Sylv]

    # set up steady state input vector
    theta0 = np.hstack([XXbar, XXbar, XXbar, YYbar, YYbar, ZZbar, ZZbar])

    ## RUN AN INITIAL SIMULATION OF RAMODEL TO GET PATHS FOR K & L

    # Find linear coefficients

    # find the derivatives matrices
    [AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM] = \
        LinApp_Deriv(Modeldyn, SIMparams, theta0, nXX, nYY, nZZ, logX)

    # set value for NN    
    NN = np.array([[rho_A]])
        
    # find the policy and jump function coefficients
    PP, QQ, RR, SS = \
        LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,NN,ZZbar,Sylv)
    #print ('P: ', PP)
    #print ('Q: ', QQ)
    #print ('R: ', RR)
    #print ('S: ', SS)\
        
    # run simulation

    # put SS values and starting values into numpy vectors
    XX0 = np.array([relK0OLG*KbarRA])

    # history of ZZ has no stochastic shocks
    ZZhist = np.zeros((T+1,nZZ))
        
    # simulate the model
    Khist, Lhist, zAhist, Yhist, whist, rhist, Fhist, Chist, Ihist \
        = generatehist(XX0, ZZhist, XXYYbar, logX, PP, QQ, RR, SS, T, \
        SIMparams)
            
    Histlist = [Khist, Lhist, zAhist, Yhist, whist, rhist, Fhist, Chist, Ihist]
            
    return Histlist, RAparams, RAbars
