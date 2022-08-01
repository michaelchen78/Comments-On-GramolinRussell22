import math
import numpy as np
import pandas as pd

'''
data = {
    "Q^2":[0.28,0.3,0.3,0.36,0.49,0.57,0.6,0.62,0.79,0.93,1,1.05,1.3,1.38,1.6,2,2,2.2,2.98],
    "G_E":[0.973,0.959,0.974,0.967,0.933,0.915,0.94,0.922,0.92,0.848,0.881,0.884,0.867,0.873,0.849,0.784,0.81,0.79,0.725]
}
'''

data = {
    "Q^2":[0.28,0.3, 0.36,0.49,0.57,0.6,0.62,0.79,0.93,1,1.05,1.3,1.38,1.6,2,2.2,2.98],
    "G_E":[0.973,0.974,0.967,0.933,0.915,0.94,0.922,0.92,0.848,0.881,0.884,0.867,0.873,0.849,0.784,0.79,0.725]
}

dx = -0.02  # based on the difference between 0.28 and 0.30
extrapolateTo = 0


n = 3  # arbitrarily selected number of derivatives
h_j = [-0.004, -0.008, -0.012, -0.016]  # defined based off of n and dx. the length must be n+1, and they must sum to dx.

decimalPlaces = 2


'''
n = 8
h_j = [-0.002, -0.004, -0.006, -0.008, -0.010, -0.012, -0.014, -0.016, -0.018]
'''


'''create bold H matrix and calculate its inverse'''
boldH = np.empty(shape=(n, n), dtype=float)
for i in range(n):
    for j in range(n):
        boldH[i][j] = ((-1*h_j[i+1])**(j+1))/(math.factorial(j+1))
#print(boldH)
boldHInv = np.linalg.inv(boldH)

'''formulate vector bold J'''
simpleArray = []
for i in range(n): simpleArray.append(1)
boldJ = np.empty(shape=(1, n), dtype=float)
boldJ[0] = simpleArray
#boldJ = np.array(simpleArray)
boldJ = np.transpose(boldJ)

'''form initial bold b and x values'''
initialBoldB = np.empty(shape=(1, len(data["G_E"])), dtype=float)
initialBoldB[0] = data["G_E"]
initialBoldB = np.transpose(initialBoldB)  # no matrix.transpose?
initialBoldX = list(data["Q^2"])


'''RBFs'''
def gaussianKernelRBF(x, x_j, c):
    return math.exp(-(abs(x-x_j)**2)/(c**2))
def gaussianFirstIntegrationRBF(x, x_j, c):
    return (c/2.0)*math.sqrt(math.pi)*math.erf((x-x_j)/c)
def gaussianSecondIntegerRBF(x, x_j, c):
    return (0.5)*\
           ((c**2)
            *math.exp(
                       -1.0*((x-x_j)**2)/(c**2)
                   )
            + c*math.sqrt(math.pi)*(x-x_j)*math.erf((x-x_j)/c))


'''iteration'''
boldB = initialBoldB
boldX = initialBoldX
for x in range(int(abs((extrapolateTo-data["Q^2"][0])/dx))):


    '''create matrix bold big phi and calculate its inverse'''
    boldBigPhi = np.empty(shape=(len(boldX), len(boldX)), dtype=float)  # this is possible weak point, because data is not uniform, this is off?
    # print(givenValuesOfX)
    for idx, x in enumerate(boldX):
        for jdx, y in enumerate(boldX):
            if (y != x):
                # print(idx, " ",jdx)
                boldBigPhi[idx][jdx] = gaussianKernelRBF(x, y, 0.1)
    # print(boldPhi)
    boldBigPhiInv = np.linalg.inv(boldBigPhi)

    '''create list of bold small phi'''
    listOfSmallBoldPhis = np.empty(shape=(n+1, len(boldX)), dtype=float)
    for idx, h in enumerate(h_j):
        for jdx, x in enumerate(boldX):
            listOfSmallBoldPhis[idx][jdx] = gaussianKernelRBF(x, h, 0.1)  # why is it abs value outside then inside separate?
    listOfSmallBoldPhis = np.delete(listOfSmallBoldPhis, (0), axis=0)



    '''perform calculations'''
    # calculate bold alpha
    boldAlpha = np.matmul(boldBigPhiInv,boldB)

    # calculate bold C
    boldC = np.matmul(listOfSmallBoldPhis, boldAlpha)
    print("C", boldC)

    # calculate bold D
    boldD = np.matmul(boldHInv, -1 * boldC - boldB[0] * boldJ)
    print("D", boldD)

    # calculate f(x_N+dx)
    #print(boldB[0])
    #print(boldB[0].item())
    fxNew = boldB[0].item() #weird pass by value/ref
    for i in range(len(boldD)):
        fxNew += boldD[i][0]*((dx**(i+1))/math.factorial(i+1)) #scalar or something? other fixed error?

    #renew renewable bold b and bold x
    print("boldB before", boldB)
    boldB = np.insert(boldB, 0, fxNew)
    boldB = np.delete(boldB, -1)
    print("boldB after",boldB)

    print("boldX before",boldX)
    boldX.insert(0, round(boldX[0] + dx, decimalPlaces))
    boldX = boldX[:-1]
    print("boldX after",boldX)

'''

useless
  # create smallBoldPhi1
    smallBoldPhi1 = []
    for i in initialBoldX: smallBoldPhi1.append(
        gaussianSecondIntegerRBF(initialBoldX[len(initialBoldX) - 1] - i, startValue, 0.4))
    # print(smallBoldPhi1)
    
'''



