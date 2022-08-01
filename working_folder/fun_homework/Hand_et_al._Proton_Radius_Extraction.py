import matplotlib.pyplot as plt
from lmfit.models import SineModel, DampedHarmonicOscillatorModel, PolynomialModel
from matplotlib.pyplot import show

import numpy as np
from numpy import exp, loadtxt, pi, sqrt
from numpy import exp, sin, linspace, random
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq, curve_fit
import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit import Model
from lmfit import models


lmFitGraphsOn = True



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

# make a new list of std deviations using error propogation
sigmaPrime = []
for i in range(len(xPrime)):
    sigmaTop = np.sqrt(sigmaTrim[i]**2 + sigmaTrim[i+1]**2)
    stdDev = sigmaTop / abs(xTrim[i]-xTrim[i+1])
    sigmaPrime.append(stdDev)
sigmaPrime = np.asarray(sigmaPrime)





''' lmfit '''
# simple linear fit
def linearModel(params, x, data, uncertainty):
    m = params['m']
    b = params['b']
    model = x*(m) + b
    return (data-model)/uncertainty**2
params = Parameters()
params.add('m', value=1)
params.add('b', value=0)

out = minimize(linearModel, params, args=(xOg, yOg, sigma))
print(out.params.pretty_print())





def increasingSineWave(x, amp, phaseShift, freq, growth):
    return amp * np.sin(x*freq + phaseShift) * exp(-x*x*growth)
def linear(x, slope, intercept):
    return x*(slope) + intercept


# linear fit
lModel = Model(linear)
lParams = lModel.make_params(slope=-1, intercept=0)
lResult = lModel.fit(yOg, lParams, x=xOg, weights=sigma)

print("linear model: \n", lResult.fit_report())

dely = lResult.eval_uncertainty(signa=3)
plt.plot(xOg, yOg, 'o')
plt.plot(xOg, lResult.init_fit, '--', label='initial fit')
plt.plot(xOg, lResult.best_fit,'-', label = 'best fit')
plt.fill_between(xOg, lResult.best_fit-dely, lResult.best_fit+dely, color = "#ABABAB", label='3-$\sigma$ uncertainty band')
plt.legend()
if(lmFitGraphsOn): plt.show()


# sinusoidal fit
sModel = SineModel()
# sParams = sModel.make_params(amp=1, freq = 50, shift = 0)
sParams = sModel.guess(yOg, x=xOg)
sResult = sModel.fit(yOg, sParams, x=xOg,weights=sigma)

print("sine model, \n", sResult.fit_report())

# dely = sResult.eval_uncertainty(signa=3)
plt.plot(xOg, yOg, 'o')
plt.plot(xOg, sResult.init_fit, '--', label='initial fit')
plt.plot(xOg, sResult.best_fit,'-', label = 'best fit')
# plt.fill_between(xOg, sResult.best_fit-dely, sResult.best_fit+dely, color = "#ABABAB", label='3-$\sigma$ uncertainty band')
plt.legend()
if(lmFitGraphsOn): plt.show()


# damped sine fit
dModel = DampedHarmonicOscillatorModel()
dParams = dModel.guess(yOg, x=xOg)
dResult = dModel.fit(yOg, dParams, x=xOg,weights=sigma)

print("damped sine model: \n", dResult.fit_report())

# dely = dResult.eval_uncertainty(signa=3)
plt.plot(xOg, yOg, 'o')
plt.plot(xOg, dResult.init_fit, '--', label='initial fit')
plt.plot(xOg, dResult.best_fit,'-', label = 'best fit')
# plt.fill_between(xOg, dResult.best_fit-dely, dResult.best_fit+dely, color = "#ABABAB", label='3-$\sigma$ uncertainty band')
plt.legend()
if(lmFitGraphsOn): plt.show()


# polynomial fit
pModel = PolynomialModel(degree = 2)
pParams = pModel.guess(yOg, x=xOg)
pResult = pModel.fit(yOg, pParams, x=xOg,weights=sigma)

print("polynomial model: \n", pResult.fit_report())

# dely = dResult.eval_uncertainty(signa=3)
plt.plot(xOg, yOg, 'o')
plt.plot(xOg, pResult.init_fit, '--', label='initial fit')
plt.plot(xOg, pResult.best_fit,'-', label = 'best fit')
# plt.fill_between(xOg, dResult.best_fit-dely, dResult.best_fit+dely, color = "#ABABAB", label='3-$\sigma$ uncertainty band')
plt.legend()
if(lmFitGraphsOn): plt.show()

polynomial = [0.12009786,1.13946492, -4.25640954, 8.01655162,-8.13269234,4.46476938, -1.37961438, 1.14473762]
polynomial = [0.01086948,-0.12210569, 0.99673977]
polyDeriv = np.polyder( polynomial, m=1)
eval = np.polyval(polyDeriv, 0)
print("EVAL: ", eval)

plt.close()


# extrapolating the derivative two ways

# first: take all the numerical derivatives, and extrapolate that data:
out2 = minimize(linearModel, params, args=(xPrime, yPrime, sigmaPrime))
print("Extrapolate derivatves:", out2.params.pretty_print())
plt.figure()
plt.plot(xTrim, yTrim, "o", label="data")
plt.plot(xPrime, yPrime, "o", label="numerical derivatives midway")
def outputF(x): return x*out2.params['m'] + out2.params['b']
x = np.linspace(-0.1,3.1, 1000)
plt.plot(x, outputF(x), '-', label="best fit line")
plt.legend()
plt.show()


# first, part two: remove the obvious outliers
print(xPrime, yPrime, sigmaPrime)
badIndex = [4,5,7,8]
xPrimeNew = np.delete(xPrime, badIndex)
yPrimeNew = np.delete(yPrime, badIndex)
sigmaPrimeNew = np.delete(sigmaPrime, badIndex)
print(xPrimeNew, yPrimeNew, sigmaPrimeNew)

out3 = minimize(linearModel, params, args=(xPrimeNew, yPrimeNew, sigmaPrimeNew))
print("Extrapolate derivatves with outliers removed:", out3.params.pretty_print())
plt.close()
plt.figure()
plt.plot(xTrim, yTrim, "o", label="data")
plt.plot(xPrimeNew, yPrimeNew, "o", label="numerical derivatives midway")
x = np.linspace(-0.1,3.1, 1000)
def outputF2(x): return x*out3.params['m'] + out3.params['b']
plt.plot(x, outputF2(x), '-', label="best fit line")
plt.legend()
plt.show()

plt.close()
# linear fit
lModel = Model(linear)
lParams = lModel.make_params(slope=-1, intercept=0)
lResult = lModel.fit(yPrimeNew, lParams, x=xPrimeNew, weights=sigmaPrimeNew)

print("linear model: \n", lResult.fit_report())

dely = lResult.eval_uncertainty(signa=3)
plt.plot(xPrimeNew, yPrimeNew, 'o')
plt.plot(xPrimeNew, lResult.init_fit, '--', label='initial fit')
plt.plot(xPrimeNew, lResult.best_fit,'-', label = 'best fit')
plt.fill_between(xPrimeNew, lResult.best_fit-dely, lResult.best_fit+dely, color = "#ABABAB", label='3-$\sigma$ uncertainty band')
plt.legend()
plt.show()


# second: extrapolate the data, then take the derivative

# positions to inter/extrapolate
x = np.linspace(0.28, -0.02, 50)

# spline order: 1 linear, 2 quadratic, 3 cubic ...
order = 3

# do inter/extrapolation
s = InterpolatedUnivariateSpline(xTrim, yTrim, k=order)
y = s(x)
plt.figure(1, dpi=120)
plt.plot(x, y, label="Function")

xPrime, yPrime = D(x, y)
plt.plot(xPrime, yPrime, label="Derivative")

for i in range(len(xPrime)):
    print("x: ", xPrime[i], "y': ", yPrime[i])