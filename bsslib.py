import numpy as np
import sympy as sym
import scipy.stats as stat
import math

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
    #print("original data: ", data)
    # initialize an identity matrix as your intial mixing matrix
    mixingA = np.identity(len(data)) # dimension = m, might need to be n instead, but i think this is right
    #print("original mixing matrix: ", mixingA)
    # TODO: replace this with a better stop criterion, this is just for testing
    i = 0
    while (i < 200):
        # calculate source signals with current mixing matrix
        signalA = unmix_signals(mixingA, data)
        #print("signal: ", signalA)
        # calculate the mixing matrix gradient
        # gradA = -A(zs^T-I)
        # z = (log P(s))'
        signals = [signal for signal in signalA] # [] of si
        gradA = signal_gradient(mixingA, data)
        # update the mixing matrix with the gradient multiplied by step size
        step_size = .1
        mixingA += step_size*gradA
        #print("unmixed: ", unmix_signals(mixingA, data))
        i += 1
    # recover the original sources
    recovered_data = unmix_signals(mixingA, data)
    return mixingA, recovered_data


# this follows slide 17 from the lecture notes, but it's probably not entirely correct, TODO: fix it
# z is the derivative of the log pdf of the generalized gaussian, that's what we're missing
# how are we supposed to get P(s_i) from the generalized Gaussian?  how does that relate to the data
# we have?
def signal_gradient(mixingA, mixedA):
    q = 1 # since we're assuming laplace distribution
    #psis = []
    #psi_funcs = []
    # TODO: THE COLUMNS OF THE MIXING MATRIX ARE THE SOURCE VECTORS
    # TODO: therefore that's what you plug in to the equations below
    # TODO: I think the algorithm below is right except the currying may not be entirely correct
    # TODO: and need to plug in the columns of                                                                                                                                                   the mixing matrix as the source vectors
    # TODO: but I'm to tired to work on it tonight
    # TODO: but now we're not using the signal matrix?
    # for i in range(len(signalA)):
    #     # something's probably not right here, but we'll work on that...
    #     # sym lets us do derivatives automatically
    #     # x = sym.Symbol('x')
    #     q = 1 # we're supposed to assume Laplace distribution, for which q=1
    #
    #     # P(s_i) follows the generalized gaussian, TODO: but what are the params of that function? must be
    #     # different for each si
    #     # TODO: it's sympy throwing an error on the next line,
    #     # "cannot determine truth value of Relational", meaning something's not in the right format
    #     # pdf is just a regular function
    #     # TODO: let's try making the zfunc function, then applying to vector s to get z
    #     #zfunc = lambda x: sym.diff(sym.log(stat.gennorm.pdf(x,q)))
    #     psi_func = lambda x: stat.gennorm.pdf(x,q) # TODO: NOTE: these functions are all the same for
    #                                         # TODO: every si, only the parameter x changes, so there's
    #                                         # TODO: no need to curry when multiplying them
    #     print(psi_func)
    #     psi_funcs.append(psi_func)
    #
    #     #TODO: first draw pdfs from here
    #
    #     psi = 0 # TODO: uncomment line below and remove this
    #     psi = stat.gennorm.pdf(signalA, q)# q is equivalent to beta
    #     print("Psi: ", psi)
    #     psis.append(psi)

    # TODO: this function isn't right, always returns 0
    # TODO: split up lambda, it's trying to differentiate a scalar which is of course 0
    #integ_z = lambda s: sym.log(np.prod([stat.gennorm.pdf(si,q) for si in s]))
    #print("test of integ_z: ", integ_z(mixingA))
    #x = sym.symbols('x')
    # z = sym.diff(integ_z(x),x)
    sigma = math.sqrt(.5)
    zfunc = lambda x: -math.sqrt(2)*x/sigma*abs(x) # TODO: put math derivation in notebook

    signalA = unmix_signals(mixingA, mixedA)

    # NOTE: z is a 2xn matrix, like x and s
    z = np.array([np.array([zfunc(xs) for xs in x]) for x in signalA])
    #z = np.array([np.array([-np.sign(xs) for xs in x]) for x in signalA])
    z = -np.sign(signalA)
    #z = np.apply_over_axes(zfunc,signalA,(0,1))

    #print("test of z: ", z)
    # THIS CAN'T BE MIXINGA, because it's a diagonal matrix and so result will always be scaled std basis vectors
    #z_of_st = np.apply_along_axis(z, 1, np.transpose(mixingA)) #TODO: this should probably be mixingA, but then we're not using signalA at all, which doesn't make sense
    z_of_st = np.matmul(z, np.transpose(signalA))
    #print("zst", z_of_st)
    #ps_func = lambda xs: func_prod(psi_funcs, xs)# make the product of all the psi functions a function
    #zfunc1 = lambda x: sym.diff(sym.log(ps_func(x)))
    #zfunc = lambda xs: zfunc1(ps_func(xs))
    # multiply all the s_i together to get s
    #ps = np.prod(np.asarray(psis))
    #print("ps: ", ps)
    # then take the log, and derive
    #log_ps = np.log(ps)
    #z = log_ps.diff(x)
    #zst = np.matmul(z, np.transpose(signalA))
    #zst = zfunc(np.transpose(signalA))
    #print("zst: ", zst) # TODO: it's 0?  why?
    #return -np.matmul(mixingA, zst - np.identity(np.shape(zst)[0]))
    ident = np.identity(np.shape(z_of_st)[0])
    temp = z_of_st + ident
    #print("mixingA: ", mixingA)
    #print("temp: ", temp)
    # it's necessary to convert because sympy turns it into an object array
    neg_gradA = np.matmul(mixingA.astype(float), temp.astype(float))
    print("neg_gradA: ", neg_gradA)
    return -neg_gradA


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


