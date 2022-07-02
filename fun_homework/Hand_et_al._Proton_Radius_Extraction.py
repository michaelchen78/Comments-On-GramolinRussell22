import matplotlib.pyplot as plt
from matplotlib.pyplot import show

import numpy as np
from numpy import exp, loadtxt, pi, sqrt
from numpy import exp, sin, linspace, random
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq
import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit import Model


''' DATA '''
xOg = [0.28,0.3,0.3,0.36,0.49,0.57,0.6,0.62,0.79,0.93,1,1.05,1.3,1.38,1.6,2,2,2.2,2.98]
xOg = np.asarray(xOg)
yOg = [0.973,0.974,0.959, 0.967,0.933,0.915,0.94,0.922,0.92,0.848,0.881,0.884,0.867,0.873,0.849,0.81, 0.784,0.79,0.725]
yOg = np.asarray(yOg)
sigma=[0.014,0.01,0.006,0.04,0.009,0.037,0.007,0.01,0.037,0.034,0.009,0.009,0.025,0.036,0.004,0.013,0.024,0.006,0.021]
sigma = np.asarray(sigma)

# makes the data a cartesian product
xTrim = np.delete(xOg, 1)
xTrim = np.delete(xTrim, 14)
yTrim = np.delete(yOg, 2) # 1 or 2
yTrim = np.delete(yTrim, 14) # 14 or 15
sigmaTrim = np.delete(sigma, 2)
sigmaTrim = np.delete(sigmaTrim,14)

# makes a list of numerical derivatives. has one less point. using trimmed since the divide by 0.
def D(xList, yList):
    yPrime = np.diff(yList)/np.diff(xList)
    xPrime = []
    for i in range(len(yPrime)):
        temp = (xList[i+1] + xList[i])/2
        xPrime = np.append(xPrime, temp)  # np?
    return xPrime, yPrime
xPrime, yPrime = D(xTrim, yTrim)


''' lmfit '''
# simple linear fit
def linearModel(params, x, data, uncertainty):
    m = params['m']
    b = params['b']
    model = x*(m) + b
    return (data-model)/uncertainty
params = Parameters()
params.add('m', value=1)
params.add('b', value=0)

out = minimize(linearModel, params, args=(xOg, yOg, sigma))
print(out.params.pretty_print())

out2 = minimize(linearModel, params, args=(xPrime, yPrime, 1))
print(out2.params.pretty_print())



def increasingSineWave(x, amp, phaseShift, freq, growth):
    return amp * np.sin(x*freq + phaseShift) * exp(-x*x*growth)


sModel = Model(increasingSineWave())










