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
plt.show()

polynomial = [0.12009786,1.13946492, -4.25640954, 8.01655162,-8.13269234,4.46476938, -1.37961438, 1.14473762]
polynomial = [0.01086948,-0.12210569, 0.99673977]
polyDeriv = np.polyder( polynomial, m=1)
eval = np.polyval(polyDeriv, 0)
print("EVAL: ", eval)

plt.close()