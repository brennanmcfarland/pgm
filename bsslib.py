import numpy as np
import sympy as sym
import scipy.stats as stat

# assumptions about the data matrix:
# it's an mxn matrix where m is number of sources and n is number of samples


# optimizes the mixing matrix
# initialize an identity matrix as your intial mixing matrix
# calculate source sound tracks with current mixing matrix
# calculate gradient of mixing matrix
# update mixing matrix with its gradient multiply with a step size (if your method cannot converge, try a smaller step size)
# go to step 2 until stop criteria fulfilled
# calculate recovered source sound tracks with final mixing matrix
# return mixing matrix and recovered source sound tracks
def bss(data):
    print(data)
    # initialize an identity matrix as your intial mixing matrix
    mixingA = np.identity(len(data)) # dimension = m, might need to be n instead, but i think this is right
    print(mixingA)
    # TODO: replace this with a better stop criterion, this is just for testing
    i = 0
    while (i < 1):
        # calculate source signals with current mixing matrix
        signalA = unmix_signals(mixingA, data)
        print(signalA)
        # calculate the mixing matrix gradient
        # gradA = -A(zs^T-I)
        # z = (log P(s))'
        signals = [signal for signal in signalA] # [] of si
        gradA = signal_gradient(mixingA, signalA)
        # update the mixing matrix with the gradient multiplied by step size
        step_size = .1
        mixingA += step_size*gradA
        i += 1
    # recover the original sources
    recovered_data = unmix_signals(mixingA, data)
    return mixingA, recovered_data


# this follows slide 17 from the lecture notes, but it's probably not entirely correct, TODO: fix it
# z is the derivative of the log pdf of the generalized gaussian, that's what we're missing
# how are we supposed to get P(s_i) from the generalized Gaussian?  how does that relate to the data
# we have?
def signal_gradient(mixingA, signalA):
    psis = []
    psi_funcs = []
    for i in range(len(signalA)):
        # something's probably not right here, but we'll work on that...
        # sym lets us do derivatives automatically
        # x = sym.Symbol('x')
        q = 1 # we're supposed to assume Laplace distribution, for which q=1

        # P(s_i) follows the generalized gaussian, TODO: but what are the params of that function? must be
        # different for each si
        # TODO: it's sympy throwing an error on the next line,
        # "cannot determine truth value of Relational", meaning something's not in the right format
        # pdf is just a regular function
        # TODO: let's try making the zfunc function, then applying to vector s to get z
        #zfunc = lambda x: sym.diff(sym.log(stat.gennorm.pdf(x,q)))
        psi_func = lambda x: stat.gennorm.pdf(x,q)
        print(psi_func)
        psi_funcs.append(psi_func)

        #TODO: first draw pdfs from here

        psi = 0 # TODO: uncomment line below and remove this
        psi = stat.gennorm.pdf(signalA, q)# q is equivalent to beta
        print("Psi: ", psi)
        psis.append(psi)
    ps_func = lambda xs: func_prod(psi_funcs, xs)# make the product of all the psi functions a function
    zfunc1 = lambda x: sym.diff(sym.log(ps_func(x)))
    zfunc = lambda xs: zfunc1(ps_func(xs))
    # multiply all the s_i together to get s
    ps = np.prod(np.asarray(psis))
    print("ps: ", ps)
    # then take the log, and derive
    log_ps = np.log(ps)
    #z = log_ps.diff(x)
    #zst = np.matmul(z, np.transpose(signalA))
    zst = zfunc(np.transpose(signalA))
    print("zst: ", zst) # TODO: it's 0?  why?
    return -np.matmul(mixingA, zst - np.identity(np.shape(zst)[0]))


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


# implement this one first to make testing the second easier
# should return a matrix in size 2×n, where n is the quantity of data points specified in argument of function call.
    # generate a 2×n matrix with random elements following standard Laplace distribution
    # use given mixing matrix with random matrix generate in first step compose the mixture
def syntheticDataGenerate(mixingA, nsamples):
    randomA = random_matrix(2, nsamples)
    mixedA = mix(randomA, mixingA)
    return mixedA


def random_matrix(xdim, ydim):
    randomA = np.random.laplace(size=(xdim,ydim))
    return randomA


def mix(matrix, mixing_matrix):
    return np.dot(mixing_matrix, matrix)


