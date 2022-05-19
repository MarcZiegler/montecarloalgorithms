# importing the required module
import sys

import matplotlib.pyplot as plt
import random as rdm
import math as math
import numpy as np
import sobol as sob

#Configuration Variables
SEED = 1337
FUNC = lambda input: [math.sin(x) for x in input]
BASE = 2 #Halton sequence base
DIM = 5
SKIP = 0
REALAREA = 2
DOMAIN = [(0, math.pi) for _ in DIM]
N = [i*100 for i in range(1, 50)]
TRIALS = 10

def GeneratePrimes(n):
    N = n * int(math.log(n) + math.log(math.log(n)))
    a = range(2, N)
    for i in range(int(n ** .5) + 1):
        a = list(filter(lambda x: x % a[i] or x == a[i], a))
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

def GenerateHaltonSequence(N, k):
    primes = GeneratePrimes(k + 1)
    seq = []
    for i in range(k):
        gen = GenerateHaltonGenerator(primes[i])
        seq += [[next(gen) for _ in range(N)]]
    return seq

def GenerateSobolSequences(N ,k):
    return sob.sample(k, N, SKIP)

def GenerateRandomSequence(N ,k):
    return [[rdm.random() for _ in N] for _ in k]

def CalculateDomainVolume(dom):
    return math.prod([x[1] - x[0] for x in dom])

def MDMCIntegralError(generator):
    averages, lbounds, ubounds = [], [], []
    for n in N:
        avgSum, maxE, minE = 0.0, 0.0, sys.float_info.max
        for _ in range(TRIALS):
            samples = generator(n, DIM)
            samples = [[FUNC(math.dist(DOMAIN[dindex][sindex])*samples[dindex][sindex]) for sindex in range(n)] for dindex in range(DIM)]
            error = math.fabs(REALAREA - CalculateDomainVolume(DOMAIN) * math.fsum(samples) / n)
            avgSum += error
            maxE = max(error, maxE)
            minE = min(error, minE)
        curAvg = avgSum / TRIALS
        averages += [curAvg]
        lbounds += [curAvg - minE]
        ubounds += [maxE - curAvg]
    return averages, [lbounds, ubounds]

def MDRiemannIntegralError():
    return []

if __name__ == '__main__':
    print(GeneratePrimes(5))
    print(GenerateHaltonSequence(5, 5))
    print(GenerateSobolSequences(5, 5))
    print(np.shape(GenerateHaltonSequence(5, 5)))
    print(np.shape(GenerateSobolSequences(5, 5)))
    print(MDMCIntegralError(GenerateSobolSequences))