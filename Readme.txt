This repository contains the code used in the working paper, "An Overlapping-Generations Model with Perfect Foresight Solved by the Sequential Recalibration Algorithm."  It contains the following Python files.


The key executable files are:

OLGfindSS.py - This is the initial program that must be run.  It calls subroutines found in SSfuncs.py.  The user needs to specify the parameter values for the OLG model, including the number of ability types per cohort, J.  It reads values for labor productivity over ages and ability types from the file, amat.xlsx.  It prints and saves a .png figure showing the steady state lifecycle behavior of the households.  It also stores the steady state values for the model variables and the parameter values in a .pkl file.

AKmethod.py - This runs the AKm method, solving the model using the traditional Auberbach-Kotlikoff algorithm.  It calls subroutines found in chainback.py.  The user needs to specify the value of J and the length of the transition path, T.  The program then reads in the steady state values and parameters from a previously saved .pkl file.  The user may also change the various options used in the AK algorithm, including the dampening parameter and its behavior.  The program then iterates and finds the transition path, which it saves along with information about the time-path iterations to a .pkl file.  It also prints and saves two figures to .png files:  1) a plot of the transition paths for aggregate consumption, aggregate capital, aggregate labor, GDP, the rental rate on capital, and the wage rate; and 2) the transition paths for aggregate capital over the iterations of the algorithm.

AMFmethod,py - This is parallel to AKmethod.py, but solves for the transition path using the AMF method.

SRAmethod.py - This is parallel to AKmethod.py, but solves for the transition path using the SRA method.

Hybmethod.py - This is similar to AKmethod.py.  However, rather than using one of three simple guesses for the initial time paths of aggregate capital and labor, the program reads the results from the AMF or SRA method and uses those final transitions paths as the initial guesses.  It solve using the AK method.

Accuracy.py and AccuracyHyb.py - These load data from the appropriate .pkl files and then calculate and report measures of accuracy for the AMF and SRA methods relative to the AK solution.


The following are support files that contain subroutines used by the files above.

chainback.py - This contains subroutines needed to the solve for the lifecycle behavior of the various households in the OLG model.  It is used by all three methods programs above.

SSfuncs.py - This contains subroutines needed to fine the steady state of the OLG model.

LinApp.py - This is a DSGE model solving package.  This repository contains the most recent version as of late 2022.  The package is archived and updates as needed at the Github url, https://github.com/kerkphillips/DSGE-Utilities, in the subdirectory, Linear_Approximation.

RAmodel.py and RArecalibrate.py - These contain subroutines needed to recalibrate the RA model used in the SRA algorithm.


Additional files include:

amat.xlsx - This is an Excel data file read in when calculating the steady state.  It contains the labor productivity value, a_sj, for each age and labor type.

OLG_SR_20xxxxx.pdf - The most recent version of the working paper.

Readme.txt - This file.