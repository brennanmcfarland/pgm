import numpy as np
import sympy as sym
import scipy.stats as stat
import math
import matplotlib.pyplot as plt


# performs blind source separation on the data
# optimizes the mixing matrix and applies it to the data to recover the sources
def bss(data):
    # initialize an identity matrix as your initial mixing matrix
    mixingA = np.identity(len(data))
    threshold = 100  # stop gradient ascent when gradient norm falls below this
    gradient_norms = []
    iter_count = 0
    while True:
        gradA = mixing_matrix_gradient(mixingA, data)
        # update the mixing matrix with the gradient multiplied by step size
        step_size = .1
        mixingA += step_size*gradA
        gradient_norm = np.linalg.norm(gradA)
        gradient_norms.append(gradient_norm)
        if iter_count%20 == 0:
            print("gradient norm: ", gradient_norm)
            print("mixing matrix: ", mixingA)
        iter_count += 1
        if np.linalg.norm(gradA) < threshold: break
    # recover the original sources
    recovered_data = unmix_signals(mixingA, data)
    plt.plot(range(len(gradient_norms)), gradient_norms)
    plt.show()
    return mixingA, recovered_data


# this follows slide 17 from the lecture notes
# given the estimated mixing matrix and mixed signals, determine the gradient of the mixing matrix
def mixing_matrix_gradient(mixingA, mixedA):
    signalA = unmix_signals(mixingA, mixedA)
    z = -np.sign(signalA)
    z_of_st = np.matmul(z, np.transpose(signalA))
    ident = np.identity(np.shape(z_of_st)[0])
    temp = z_of_st + ident
    gradA = -np.matmul(mixingA.astype(float), temp.astype(float))
    return gradA


def func_prod(lambs, xs):
    return func_prod_helper(lambs[0], lambs[1:], xs)


def func_prod_helper(lamb, lambs, xs):
    if len(lambs) == 0:
        return 1
    else:
        return lamb(xs[0])*func_prod_helper(lambs[0], lambs[1:], xs[1:])


# given mixing matrix and mixed signals, unmix the signals
# s = A^-1x
def unmix_signals(mixingA, mixedA):
    signalA = np.dot(np.linalg.inv(mixingA), mixedA)
    return signalA


# generate a random data matrix and apply the mixing matrix
def syntheticDataGenerate(mixingA, nsamples):
    randomA = random_matrix(2, nsamples)
    mixedA = mix(randomA, mixingA)
    return mixedA


def random_matrix(xdim, ydim):
    randomA = np.random.laplace(size=(xdim, ydim))
    return randomA


def mix(matrix, mixing_matrix):
    return np.dot(mixing_matrix, matrix)


