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
    # new textbook page 425
    # scipy multivariate normal (the denominator part factors out - normalizing constant is also sum of gaussians)
    posteriors = []
    for n in range(len(data)):
        posteriors.append([])
        for k in range(len(gmm)):
            p_k_xn = multivariate_normal.pdf(x[n], mean=gmm[k]['mean'], cov=gmm[k]['covariance']) # what to do about determinants?  do those cancel too?
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
def maximization(posterior, data, gmm):
    print(gmm)
    x = data
    gmm = maximization_mean(posterior, data, gmm)
    # NOTE: may be reusing k for both number of classes and dimensions, elsewhere too?
    # there's 3 levels of nesting: which covariance matrix itself (k),
    # the dimension of the datapoints (d),
    # and the dimension of the covariance (c)
    # but how to get the individual vals of the covariance matrix?
    covariances = []
    for k in range(len(gmm)):
        covariances_k = []
        for d in range(len(data[0])):
            #covariance = 0
            covariance_0 = [0,0]
            covariance_1 = [0,0]
            for n in range(len(data)):
                # covariance += multivariate_normal.pdf(x[n], mean=gmm[k]['mean'], cov=gmm[k]['covariance'])
                covariance_0 = [c + () for c in covariance_0]
                covariance_1 = [c + () for c in covariance_1]
            covariances_k.append(covariance)
        # TODO: this is not right, just copying the values, but why is there that extra level of nesting?
        # of course this doesn't even work for testing because the matrix is not invertible
        cov_tmp = []
        cov_tmp.append(copy.deepcopy(covariances_k))
        cov_tmp.append(copy.deepcopy(covariances_k))
        #gmm[k]['covariance'] = np.array(cov_tmp)
        gmm[k]['covariance'] = np.array(copy.deepcopy(covariances_k))
        covariances.append(covariances_k)
    print(covariances)
    print(gmm)
    # TODO: actually update covariance and class prior
    return gmm
    # n['covariance'] = covariance[n] # 0 = [np.sum(p_n_k[0]*x[n]/p_k) for n in range(len(data))]


# update just the mean for each class
# u[k] = Σ[n] p[n,k]x^(n)/p[k]
def maximization_mean(posterior, data, gmm):
    means = []
    for k in range(len(gmm)):
        # means.append[]
        mean = 0
        p_k = 0
        for n in range(len(data)):
            x_n = data[n]
            p_n_k = posterior[n][k]
            p_k += p_n_k
            mean += p_n_k*x_n
        mean /= p_k
        means.append(mean)
        gmm[k]['mean'] = copy.deepcopy(mean)
    # print(means)
    return gmm


# using sklearn's pca functionality
def dimReducePCA(data, newdimensions):
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    data = pca.transform(data)
    #TODO: needs to return reduced data, eigenvectors and eigenvalues
    # data is data components is eigenvectors, and explained variance is eigenvalues (I think?)
    print(pca.components_)
    return data, pca.components_, pca.explained_variance_
