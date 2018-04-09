import numpy as np

# calculate the expectation using the following formula:
# u[k] = Σ[n] p[n,k]x^(n)/p[k]
# p[n,k] = p(c[k]|x^(n),θ[1:K])
# the contents of data and gmm are:
# data is an array of datapoints (also arrays, with 2 elements)
# gmm = [{'mean': mu[m], 'covariance': sigma[m], 'prior': 1.0/ngmm} for m in range(ngmm)]
def expectation(data, gmm):
    # sigma = [m['covariance'] for m in gmm] # an array
    p_n_k = [m['prior'] for m in gmm]
    x = data
    p_k = np.sum(p_n_k)
    print(p_n_k) # one array of size 2
    print(x) # a whole lotta arrays of size 2
    print(p_k) # a number, 1, apparently?
    u0 = [np.sum(p_n_k[0]*x[n]/p_k) for n in range(len(data))]
    u1 = [np.sum(p_n_k[1]*x[n]/p_k) for n in range(len(data))]
    u = [u0, u1]
    return u


# update the mean, covariance, and class prior for each class?
def maximization(posterior, data, gmm):
    maximization_mean(posterior, data, gmm)
    # n['covariance'] = covariance[n] # TODO: actually update the covarianceu0 = [np.sum(p_n_k[0]*x[n]/p_k) for n in range(len(data))]


def maximization_mean(posterior, data, gmm):
    # update mean
    mean = expectation(data, gmm)
    for n in range(len(gmm)):
        gmm[n]['mean'] = mean[n]
    # update class prior
    for n in range(len(gmm)):
        gmm[n]['prior'] = posterior[n]
