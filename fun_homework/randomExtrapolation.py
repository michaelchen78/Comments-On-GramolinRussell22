import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import exp, sin, linspace, random
from scipy.optimize import leastsq
from lmfit import minimize, Parameters, Parameter


xOg = [0.28,0.3,0.3,0.36,0.49,0.57,0.6,0.62,0.79,0.93,1,1.05,1.3,1.38,1.6,2,2,2.2,2.98]
xOg = np.asarray(xOg)
yOg = [0.973,0.974,0.959, 0.967,0.933,0.915,0.94,0.922,0.92,0.848,0.881,0.884,0.867,0.873,0.849,0.81, 0.784,0.79,0.725]
yOg = np.asarray(yOg)
sigma=[0.014,0.01,0.006,0.04,0.009,0.037,0.007,0.01,0.037,0.034,0.009,0.009,0.025,0.036,0.004,0.013,0.024,0.006,0.021]

xiList = [0.28,0.3,0.36,0.49,0.57,0.6,0.62,0.79,0.93,1,1.05,1.3,1.38,1.6,2,2.2,2.98]
yiList = [0.973,0.974,0.967,0.933,0.915,0.94,0.922,0.92,0.848,0.881,0.884,0.867,0.873,0.849,0.784,0.79,0.725]
for i, s in enumerate(yiList):
    yiList[i] = s*-1

# given values
xi = np.array(xiList)
yi = np.array(yiList)


# positions to inter/extrapolate
x = np.linspace(0.28, -0.02, 50)

# spline order: 1 linear, 2 quadratic, 3 cubic ...
order = 1

# do inter/extrapolation
s = InterpolatedUnivariateSpline(xi, yi, k=order)
y = s(x)

plt.figure(1, dpi=120)

plt.plot(x, y, label="Function")

def D(xList, yList):
    yPrime = np.diff(yList)/np.diff(xList)
    xPrime = []
    for i in range(len(yPrime)):
        temp = (xList[i+1] + xList[i])/2
        xPrime = np.append(xPrime, temp) #np?
    return xPrime, yPrime

xPrime, yPrime = D(x, y)
plt.plot(xPrime, yPrime, label="Derivative")

for i in range(len(xPrime)):
    print("x: ", xPrime[i], "y': ", yPrime[i])


plt.legend()
plt.show()



xPrimeSimp, yPrimeSimp = D(xi, yi)
print(xPrimeSimp)
print(yPrimeSimp)

plt.figure(1, dpi=120)
plt.plot(xPrimeSimp, yPrimeSimp, "o", label = "Derivative Scatter Plot")


x = np.linspace(3, -0.02, 1000) #if this step is asynchrous with the existing data?
order = 1

# do inter/extrapolation
r = InterpolatedUnivariateSpline(xPrimeSimp, yPrimeSimp, k=order)
z = r(x)



plt.plot(x, z, label="Extrapolated Derivative")

print("\n\nEXTRAPOLATED DERIVATVE")
#for i in range(len(x)):
#print("x: ", x[i], "y': ", z[i])

plt.legend()
#plt.show()



'''
# example showing the interpolation for linear, quadratic and cubic interpolation
plt.figure()
for order in range(1, 5):
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)
    plt.plot(x, y, "o", label="function")
'''

x = linspace(0,100)
x2 = linspace(-10,10,101)
print(x2)
m= 0.2
print(type(m))
print("1", type(x))
print("HERE", x*m," ", x2*0.2)


def residual(params, x, data, uncertainty):
    m = params['m']
    b = params['b']
    print("2", type(x))
    model = x*float(m) + b
    return (data-model)/uncertainty

params = Parameters()
params.add('m', value=1)
params.add('b', value=0)

out = minimize(residual, params, args=(xOg, yOg, sigma))
print(out)

# <examples/doc_model_gaussian.py>
import matplotlib.pyplot as plt
from numpy import exp, loadtxt, pi, sqrt

from lmfit import Model

'''
data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]
'''
def gaussian(x,amp,cen,wid):
    return amp*exp(-(x-cen)**2/wid)
x = linspace(-10,10,101)
y = gaussian(x,2.33,0.21, 1.51) + random.normal(0,0.2,x.size)


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


gmodel = Model(gaussian)
result = gmodel.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())

plt.plot(x, y, 'o')
plt.plot(x, result.init_fit, '--', label='initial fit')
plt.plot(x, result.best_fit, '-', label='best fit')
plt.legend()
plt.show()
# <end examples/doc_model_gaussian.py>
