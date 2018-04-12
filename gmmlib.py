import copy
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import decomposition
from sklearn import datasets
# from scikit-learn.preprocessing import normalize

# calculate the expectation using the following formula:
# note: it's NOT the mean, but the probabilities of each data point being in a given class
# p[n,k] = p(c[k]|x^(n),θ[1:K])
# the contents of data and gmm are:
# data is an array of datapoints (also arrays, with 2 elements)
# gmm = [{'mean': mu[m], 'covariance': sigma[m], 'prior': 1.0/ngmm} for m in range(ngmm)]
def expectation(data, gmm):
    x = data
    posteriors = []
    for n in range(len(data)):
        posteriors.append([])
        for k in range(len(gmm)):
            p_k_xn = 0
            try:
                p_k_xn = multivariate_normal.pdf(x[n], mean=gmm[k]['mean'], cov=gmm[k]['covariance']) # what to do about determinants?  do those cancel too?
            except:
                print()
                print("xn:",x[n])
                print("mean:",gmm[k]['mean'])
                print("covar:",gmm[k]['covariance'])
                print("prior:",gmm[k]['prior'])
            p_k_xn *= gmm[k]['prior']
            posteriors[n].append(p_k_xn)
        normalizing_constant = np.sum(posteriors[n])
        # means now just go to 0
        posteriors[n] = [i/normalizing_constant for i in posteriors[n]]
    mean_normalizers = np.array(posteriors).sum(axis=0)
    for n in range(len(data)):
        for k in range(len(gmm)):
            posteriors[n] = [i/mean_normalizers[k] for i in posteriors[n]]
    return posteriors


# update the mean, covariance, and class prior for each class
def maximization(posterior, data, oldgmm):
    gmm = copy.deepcopy(oldgmm)
    #print(gmm)
    x = data
    gmm = maximization_mean(posterior, data, gmm)
    covariances = []
    priors = []
    # TODO: the means are right, but the priors and covariances
    # are both way smaller than they should be
    for k in range(len(gmm)):
        #covariances_k = np.empty(len(data))
#        covariances_k = np.zeros_like(gmm[0]['covariance'])
        covariance = np.zeros_like(gmm[k]['covariance'])#0
        prior = 0
        for n in range(len(data)):
            p_n_k = posterior[n][k]
            p_k = np.sum(posterior[n])
            prior += p_n_k/p_k

        for n in range(len(data)):
            p_n_k = posterior[n][k]
            p_k = np.sum(posterior[n])
            mat_row = np.reshape(x[n] - gmm[k]['mean'], (2,1))
            covariance += p_n_k * (mat_row * mat_row.transpose())

        print("unnormalized prior: ", prior)
        prior /= len(x)
        gmm[k]['covariance'] = covariance
        covariances.append(covariance)
        gmm[k]['prior'] = prior
        priors.append(prior)
    return gmm


# update just the mean for each class
# u[k] = Σ[n] p[n,k]x^(n)/p[k]
def maximization_mean(posterior, data, oldgmm):
    gmm = copy.deepcopy(oldgmm)
    means = []
    for k in range(len(gmm)):
        mean = np.zeros_like(gmm[k]['mean'])
        p_k = 0
        for n in range(len(data)):
            x_n = data[n]
            p_n_k = posterior[n][k]
            p_k += p_n_k
            mean += p_n_k*x_n
        mean /= p_k
        means.append(mean)
        gmm[k]['mean'] = mean
    # print(means)
    return gmm


# using sklearn's pca functionality
def dimReducePCA(data, newdimensions):
    pca = decomposition.PCA(n_components=newdimensions) # TODO: change to newdimensions?
    pca.fit(data)
    data -= np.mean(data, axis=0)
    covariance = np.matmul(data.transpose(), data)/data.shape[0]
    ev, ew = np.linalg.eig(covariance)
    pcs = pca.transform(data)
    return pcs, ew, ev
