# importing the required module
import sys

import matplotlib.pyplot as plt
import random as rdm
import math as math
import numpy as np

#Configuration Variables
SEED = 1337
FUNC = lambda theta: math.sin(theta)
REALAREA = 2
INTERVAL = (0, math.pi)
N = [i*2 for i in range(1, 50)]
TRIALS = 10

def MCIntegralError():
    averages, lbounds, ubounds = [], [], []
    for n in N:
        avgSum, maxE, minE = 0.0, 0.0, sys.float_info.max
        for _ in range(TRIALS):
            samples = []
            for _ in range(n):
                samples += [FUNC(rdm.uniform(INTERVAL[0], INTERVAL[1]))]
            error = math.fabs(REALAREA - (INTERVAL[1]-INTERVAL[0])*math.fsum(samples)/n)
            avgSum += error
            maxE = max(error, maxE)
            minE = min(error, minE)
        curAvg = avgSum / TRIALS
        averages += [curAvg]
        lbounds += [curAvg - minE]
        ubounds += [maxE - curAvg]
    return averages, [lbounds, ubounds]

def SimpsonIntegralError():
    results = []
    for n in N:
        deltaX = (INTERVAL[1]-INTERVAL[0])/n
        xk = lambda k: INTERVAL[0] + k*deltaX
        iSum = 0.0
        for i in range(1, math.floor(n/2)):
            iSum += FUNC(xk(2*i-2)) + 4*FUNC(xk(2*i-1)) + FUNC(xk(2*i))
        results += [math.fabs(REALAREA - iSum*deltaX/3)]
    return results

def RiemannIntegralError():
    results = []
    for n in N:
        deltaX = (INTERVAL[1]-INTERVAL[0])/n
        xk = lambda k: INTERVAL[0] + k*deltaX
        iSum = 0.0
        for i in range(1, n):
            iSum += FUNC(xk(i))
        results += [math.fabs(REALAREA - iSum*deltaX)]
    return results


if __name__ == '__main__':
    rdm.seed(SEED)
    mcRes = MCIntegralError()
    sRes = SimpsonIntegralError()
    rRes = RiemannIntegralError()

    # x axis values
    x = N
    # corresponding y axis values
    ymc = mcRes[0]
    ymcerror = mcRes[1]
    ys = sRes
    yr = rRes

    print(ymcerror[0])
    print(mcRes[0])
    print(ymcerror[1])
    print(np.shape(ymcerror))

    # plotting the points
    graph = plt.figure()
    mcPlot = graph.add_subplot(1,1,1)
    mcPlot.errorbar(x, ymc, yerr=ymcerror, color='tab:blue', label='MC MSE')
    sPlot = graph.add_subplot(1, 1, 1)
    sPlot.plot(x, ys, color='tab:red', label='Simpson MSE')
    sPlot = graph.add_subplot(1, 1, 1)
    sPlot.plot(x, yr, color='tab:green', label='Riemann MSE')
    plt.legend(loc='upper right')

    # naming the x axis
    plt.xlabel('Number of subintervals / samples N')
    # naming the y axis
    plt.ylabel('Absolute error')

    # giving a title to my graph
    plt.title('Absolute integration error of sin(x) over [0,pi]')

    # function to show the plot
    plt.show()
