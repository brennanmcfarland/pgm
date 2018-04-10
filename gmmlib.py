import numpy as np
from scipy.stats import multivariate_normal
# from scikit-learn.preprocessing import normalize

# calculate the expectation using the following formula:
# note: it's NOT the mean, but the probabilities of each data point being in a given class
# u[k] = Σ[n] p[n,k]x^(n)/p[k]
# p[n,k] = p(c[k]|x^(n),θ[1:K])
# the contents of data and gmm are:
# data is an array of datapoints (also arrays, with 2 elements)
# gmm = [{'mean': mu[m], 'covariance': sigma[m], 'prior': 1.0/ngmm} for m in range(ngmm)]
# TODO: the means blow up really fast, why? also the first and second entries are always the same
def expectation(data, gmm):
    # sigma = [m['covariance'] for m in gmm] # an array
    # print(gmm[0]['prior'])
    p_n_k = []
    p_n_k.append(gmm[0]['prior'])
    p_n_k.append(gmm[1]['prior'])
    # p_n_k = [m['prior'] for m in gmm]
    x = data
    p_k = [np.sum(p_n_k[0]), np.sum(p_n_k[1])]
    #print(p_n_k) # one array of size 2
    #print(x) # a whole lotta arrays of size 2
    # print(p_k) # a number, 1, apparently?
    u = []
    posterior = [[gmm[0]['prior'], gmm[1]['prior']] for n in data]

    # new textbook page 425
    # x is a vector
    # so is mu, etc

    posteriors = []
    # scipy multivariate normal (the denominator part factors out - normalizing constant is also sum of gaussians)
    for n in range(len(data)):
        posteriors.append([])
        for k in range(len(gmm)):
            p_k_xn = multivariate_normal.pdf(x[n], mean=gmm[k]['mean'], cov=gmm[k]['covariance']) # what to do about determinants?  do those cancel too?
            posteriors[n].append(p_k_xn)
        normalizing_constant = max(np.sum(posteriors[n]), .000000001) #cause it could be 0
        # means now just go to 0
        posteriors[n] = [i/normalizing_constant for i in posteriors[n]]
    return posteriors

    # for k in range(len(data[0])):
        #print(p_n_k[0])
        #print(x[0][0])
        #print(p_k[0])
        # u0 = np.sum([p_n_k[0]*x[n][0]/p_k[0] for n in range(len(data))])
        # u1 = np.sum([p_n_k[1]*x[n][1]/p_k[1] for n in range(len(data))])
        # u.append([u0, u1])
    # return u


# update the mean, covariance, and class prior for each class?
def maximization(posterior, data, gmm):
    return maximization_mean(posterior, data, gmm)
    # n['covariance'] = covariance[n] # TODO: actually update the covarianceu0 = [np.sum(p_n_k[0]*x[n]/p_k) for n in range(len(data))]


def maximization_mean(posterior, data, gmm):
    # print("old gmm: ", gmm)
    # print("posterior: ", posterior)
    # update mean

    # mean = expectation(data, gmm)
    # for n in range(len(gmm)):
    #     gmm[n]['mean'] = mean[n]
    # print("new mean: ", mean)
    # update class prior
    # for n in range(len(gmm)):
    #     gmm[n]['prior'] = posterior[n][0] # TODO: this isn't right, why is posterior a nested array?
    # print("new prior: ", posterior)
    # print("new gmm: ", gmm)
    return gmm

# NOTE: scikit learn has pca algorithms
