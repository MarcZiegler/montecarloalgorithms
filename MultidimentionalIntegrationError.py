# importing the required module
import sys

import matplotlib.pyplot as plt
import random as rdm
import math as math
import numpy as np
import sobol as sob

#Configuration Variables
SEED = 1337
FUNC = lambda input: 1 if math.fsum([math.pow(x, 2) for x in input]) <= 1 else 0
DIM = 6
SKIP = 0
REALAREA = math.pow(math.pi, 3)/6 #for hypersphere in 6 DIM
DOMAIN = (-1, 1) #Hypercube domain
N = [i*100 for i in range(20, 30)] #Number of samples
TRIALS = 1

def GeneratePrimes(n):
    no = n * int(math.log(n) + math.log(math.log(n)))
    a = range(2, no)
    for i in range(int(n ** .5) + 1):
        a = list(filter(lambda o: o % a[i] or o == a[i], a))
    return a

def GenerateHaltonGenerator(b):
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

def GenerateHaltonSequences(n, k):
    primes = GeneratePrimes(k + 5) #TODO this behaves weird
    seq = []
    for i in range(k):
        gen = GenerateHaltonGenerator(primes[i])
        seq += [[next(gen) for _ in range(n)]]
    return np.transpose(seq) #TODO make this more efficient

def GenerateSobolSequences(n, k):
    return sob.sample(k, n, SKIP)

def GenerateRandomSequences(n, k):
    return [[rdm.random() for _ in range(k)] for _ in range(n)]

def CalculateDomainVolume():
    return math.pow(DOMAIN[1] - DOMAIN[0], DIM)

def MDMCIntegralError(generator):
    averages, lbounds, ubounds = [], [], []
    for n in N:
        avgSum, maxE, minE = 0.0, 0.0, sys.float_info.max
        for _ in range(TRIALS):
            samples = generator(n, DIM)
            samples = [FUNC([DOMAIN[0]+(DOMAIN[1]-DOMAIN[0])*s for s in nn]) for nn in samples]
            error = math.fabs(REALAREA - CalculateDomainVolume() * math.fsum(samples) / n)
            avgSum += error
            maxE = max(error, maxE)
            minE = min(error, minE)
        curAvg = avgSum / TRIALS
        averages += [curAvg]
        lbounds += [curAvg - minE] #abs error
        ubounds += [maxE - curAvg] #abs error
    return averages, [lbounds, ubounds]

def MDRiemannIntegralError():
    return []

if __name__ == '__main__':
    rdm.seed(SEED)

    print(GeneratePrimes(5))
    print(GenerateHaltonSequences(6, 5))
    print(GenerateSobolSequences(6, 5))
    print(np.shape(GenerateHaltonSequences(6, 5)))
    print(np.shape(GenerateSobolSequences(6, 5)))
    #print(MDMCIntegralError(GenerateSobolSequences))

    randomRes = MDMCIntegralError(GenerateRandomSequences)
    sobolRes = MDMCIntegralError(GenerateSobolSequences)
    haltonRes = MDMCIntegralError(GenerateHaltonSequences)

    # x axis values
    x = N
    # corresponding y axis values
    yrandom = randomRes[0]
    yrandome = randomRes[1]
    ysobol = sobolRes[0]
    ysobole = sobolRes[1]
    yhalton = haltonRes[0]
    yhaltone = haltonRes[1]

    # plotting the points
    graph = plt.figure()
    randomPlot = graph.add_subplot(1, 1, 1)
    rebar = randomPlot.errorbar(x, yrandom, yerr=yrandome, ecolor='#82FF56', capsize=2.5, elinewidth=2.0, color='tab:green',
                    label='Random Sampling')
    sobolPlot = graph.add_subplot(1, 1, 1)
    sobolPlot.errorbar(x, ysobol, yerr=ysobole, ecolor='#FF878E', capsize=2.5, elinewidth=2.0, color='tab:red',
                    label='Sobol Sampling')
    haltonPlot = graph.add_subplot(1, 1, 1)
    haltonPlot.errorbar(x, yhalton, yerr=yhaltone, ecolor='#62DBFF', capsize=2.5, elinewidth=2.0, color='tab:blue',
                    label='Halton Sampling')
    plt.legend(loc='upper right')
    # naming the x axis
    plt.xlabel('Number of subintervals / samples N')
    # naming the y axis
    plt.ylabel('Absolute error')

    # giving a title to my graph
    plt.title('Absolute integration error of a 6 Dimensional Hypersphere')

    # function to show the plot
    plt.show()