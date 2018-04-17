import numpy as np


# optimizes the mixing matrix
# initialize an identity matrix as your intial mixing matrix
# calculate source sound tracks with current mixing matrix
# calculate gradient of mixing matrix
# update mixing matrix with its gradient multiply with a step size (if your method cannot converge, try a smaller step size)
# go to step 2 until stop criteria fulfilled
# calculate recovered source sound tracks with final mixing matrix
# return mixing matrix and recovered source sound tracks
def bss(data):
    # TODO: these are obviously stubs
    recovered_data = None
    optimalA = np.zeroes(2,2)
    return optimalA, recovered_data


# implement this one first to make testing the second easier
# should return a matrix in size 2×n, where n is the quantity of data points specified in argument of function call.
    # generate a 2×n matrix with random elements following standard Laplace distribution
    # use given mixing matrix with random matrix generate in first step compose the mixture
def syntheticDataGenerate(mixingmatrix, nsamples):
    randomA = random_matrix(2, nsamples)
    mixedA = np.dot(mixingmatrix, randomA)
    return mixedA


def random_matrix(xdim, ydim):
    randomA = np.random.laplace(size=(xdim,ydim))
    return randomA


