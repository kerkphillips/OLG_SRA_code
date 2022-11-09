# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 07:39:12 2022

@author: Kerk
"""

import numpy as np
import matplotlib.pyplot as plt
from LinApp import LinApp_FindSS, LinApp_Deriv, LinApp_Solve, LinApp_SSL

# create a definitions function
def Modeldefs(XXp, XX, YY, ZZ, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital holdings in next period
        X: value of capital holdings this period
        Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Y: GDP
        w: wage rate
        r: rental rate on capital
        F: transfer payments
        c: consumption
        u: utiity
    '''
    
    # unpack input vectors
    Kp = XXp[0]
    K = XX[0]
    L = YY[0]
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho_A, sigma_A, \
        ZZbar, nXX, nYY, nZZ, logX, Sylv] = params
    # find definintion values
    Y = K**alpha*L**(1-alpha)
    w = (1-alpha)*Y/L  
    r = alpha*Y/K
    F = tau*(w*L + (r-delta)*K)
    C = (1-tau)*(w*L + (r - delta)*K) + K + F - Kp
    I = Kp - (1-delta)*K
    
    return Y, w, r, F, C, I


def Modeldyn(theta0, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital in two periods
            Xp: value of capital in next period
            X: value of capital this period
            Yp: value of labor in next period
            Y: value of labor this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
        
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho_A, sigma_A, \
        ZZbar, nXX, nYY, nZZ, logX, Sylv] = params
        
    # unpack theta0
    XXpp = theta0[0 : nXX]
    XXp = theta0[nXX : 2*nXX]
    XX = theta0[2*nXX : 3*nXX]
    YYp = theta0[3*nXX : 3*nXX + nYY]
    YY = theta0[3*nXX + nYY : 3*nXX + 2*nYY]
    ZZp = theta0[3*nXX + 2*nYY : 3*nXX + 2*nYY + nZZ]
    ZZ = theta0[3*nXX + 2*nYY+ nZZ : 3*nXX + 2*nYY + 2*nZZ]
    
    L = YY[0]
    
    # find definitions for now and next period
    Y, w, r, F, C, I = Modeldefs(XXp, XX, YY, ZZ, params)
    Yp, wp, rp, Fp, Cp, Ip = Modeldefs(XXpp, XXp, YYp, ZZp, params)

    # Euler equations
    EYY = C**(-gamma)*w*(1-tau) - (chi*L**theta)
    EXX = (C**(-gamma)) - (beta*Cp**(-gamma)*(1 + (1-tau)*(rp - delta)))
    E = np.array([EYY, EXX])

    return E


def generatehist(XX0, ZZhist, XXYYbar, logX, PP, QQ, RR, SS, nobs, params):
    
    # simulate the model
    XXhist, YYhist = \
        LinApp_SSL(XX0, ZZhist, XXYYbar, logX, PP, QQ, RR, SS)
        
    # generate non-state variables
    Yhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    Fhist = np.zeros(nobs)
    Chist = np.zeros(nobs)
    Ihist = np.zeros(nobs)
    
    for t in range(0, nobs):
        Yhist[t], whist[t], rhist[t], Fhist[t], Chist[t], Ihist[t] \
            = Modeldefs(XXhist[t+1], XXhist[t], YYhist[t], ZZhist[t,:], params) 
    Khist = XXhist[0:nobs]
    Lhist = YYhist[0:nobs]
    zAhist = ZZhist[0:nobs,0]
    
    return Khist, Lhist, zAhist, Yhist, whist, rhist, Fhist, Chist, \
        Ihist


def plothist(Khist, Lhist, Yhist, Chist, whist, rhist, Fhist, Ihist, title):
    # plot
    
    fig = plt.figure()
    
    fig.suptitle(title)
        
    plt.subplot(4, 2, 1)
    plt.plot(Khist)
    plt.xticks([])
    plt.title('Capital', y=.92)
    
    plt.subplot(4, 2, 2)
    plt.plot(Lhist)
    plt.xticks([])
    plt.title('Labor', y=.92)
    
    plt.subplot(4, 2, 3)
    plt.plot(Yhist)
    plt.xticks([])
    plt.title('GDP', y=.92)
    
    plt.subplot(4, 2, 4)
    plt.plot(Chist)
    plt.xticks([])
    plt.title('Consumption', y=.92)
    
    plt.subplot(4, 2, 5)
    plt.plot(whist)
    plt.xticks([])
    plt.title('Wage', y=.92)
    
    plt.subplot(4, 2, 6)
    plt.plot(rhist)
    plt.xticks([])
    plt.title('Interest Rate', y=.92)
    
    plt.subplot(4, 2, 7)
    plt.plot(Fhist)
    plt.title('Transfers', y=.92)
    
    plt.subplot(4, 2, 8)
    plt.plot(Ihist)
    plt.title('Investment', y=.92)
    
    fig.show()