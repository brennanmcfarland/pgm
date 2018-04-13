import copy
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import decomposition
from sklearn import datasets

# calculate the expectation
# p[n,k] = p(c[k]|x^(n),θ[1:K])
def expectation(data, gmm):
    posteriors = []
    # calculate unnormalized posteriors
    for n in range(len(data)):
        posteriors.append([])
        for k in range(len(gmm)):
            p_k_xn = 0
            try:
                p_k_xn = multivariate_normal.pdf(data[n], mean=gmm[k]['mean'], cov=gmm[k]['covariance'])
            except:
                print("There was an error calculating p_k_xn.  Not sure why it does this but it messes things up")
                print("xn:",data[n])
                print("mean:",gmm[k]['mean'])
                print("covar:",gmm[k]['covariance'])
                print("prior:",gmm[k]['prior'])
            p_k_xn *= gmm[k]['prior']
            posteriors[n].append(p_k_xn)
        # normalize over n
        normalizing_constant = np.sum(posteriors[n])
        posteriors[n] = [i/normalizing_constant for i in posteriors[n]]
    # normalize the individual posteriors
    mean_normalizers = np.array(posteriors).sum(axis=0)
    for n in range(len(data)):
        for k in range(len(gmm)):
            posteriors[n] = [i/mean_normalizers[k] for i in posteriors[n]]
    return posteriors


# update the mean, covariance, and class prior for each class
def maximization(posterior, data, oldgmm):
    gmm = copy.deepcopy(oldgmm)
    gmm = maximization_mean(posterior, data, gmm)
    # calculate unnormalized priors and covariance matrices
    for k in range(len(gmm)):
        covariance = np.zeros_like(gmm[k]['covariance'])
        prior = 0
        # calculate priors
        for n in range(len(data)):
            p_n_k = posterior[n][k]
            p_k = np.sum(posterior[n])
            prior += p_n_k/p_k
        # calculate covariances
        for n in range(len(data)):
            p_n_k = posterior[n][k]
            p_k = np.sum(posterior[n])
            mat_row = np.reshape(data[n] - gmm[k]['mean'], (2,1))
            covariance += p_n_k * (mat_row * mat_row.transpose())

        #normalize the prior
        prior /= len(data)
        # update the model
        gmm[k]['covariance'] = covariance
        gmm[k]['prior'] = prior
    return gmm


# update just the mean for each class
# u[k] = Σ[n] p[n,k]x^(n)/p[k]
def maximization_mean(posterior, data, oldgmm):
    gmm = copy.deepcopy(oldgmm)
    for k in range(len(gmm)):
        mean = np.zeros_like(gmm[k]['mean'])
        p_k = 0
        for n in range(len(data)):
            x_n = data[n]
            p_n_k = posterior[n][k]
            p_k += p_n_k
            mean += p_n_k*x_n
        mean /= p_k # normalize the mean
        gmm[k]['mean'] = mean # update the model
    return gmm


# using sklearn's pca functionality
# reduce the data to newdimensions dimensions w/ PCA
# returns the principal components, eigenvectors and eigenvalues
def dimReducePCA(data, newdimensions):
    pca = decomposition.PCA(n_components=newdimensions) # TODO: change to newdimensions?
    pca.fit(data)
    data -= np.mean(data, axis=0)
    covariance = np.matmul(data.transpose(), data)/data.shape[0]
    ev, ew = np.linalg.eig(covariance)
    pcs = pca.transform(data)
    return pcs, ew, ev
