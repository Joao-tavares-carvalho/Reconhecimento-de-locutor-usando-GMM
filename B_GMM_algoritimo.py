import numpy as np
import math
import sklearn.utils.random as utils_random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy.matlib as matlib
from sklearn.datasets import make_blobs
import sklearn.utils as utils
import matplotlib.pyplot as plt


# Distancia euclidiana de um ponto (amostra) para cada ponto contido no cluster
# sample = [1xM], clusters = [KxM]
def distance(sample, clusters):
    sample_rep = np.matlib.repmat(sample, len(clusters[:, 1]), 1)
    dist = clusters - sample_rep
    # return size [1xK]
    return np.linalg.norm(dist, 2, axis=1) ** 2

#Menor destancia de um ponto a qualquer ponto no cluster
# sample = [1xM], clusters = [KxM]
def min_distance(sample, clusters):
    dist = distance(sample, clusters)
    # return scalar
    return np.min(dist)

#Indece do ponto clusters que esta mais prossimo da amostra
# sample = [1xM], clusters = [KxM]
def argmin_distance(sample, clusters):
    dist = distance(sample, clusters)
    # return scalar
    return np.argmin(dist)

#Distancia media minima de todos os pontos de dados
# samples = [NxM], clusters = [KxM]
def mean_distance(samples, clusters):
    dist = np.zeros(len(samples[:, 1]))

    for i in range(len(samples[:, 1])):
        dist[i] = min_distance(samples[i, :], clusters)

    # return scalar
    return np.mean(dist)


# samples = [MxN], n_clusters = K, max_iterations = positive scalar, clusters_init = [K,N] (optional)
def kmeans_custom(samples, n_clusters, max_iterations, clusters_init=None):
    # Some termination thresholds
    dist_thr = 1 / n_clusters
    stab_thr = 1 - 1 * 10 ** -10

    # Split into training and test set (e.g. by shuffling or by generating random indices)
    samples_shuffled = np.copy(samples)
    np.random.shuffle(samples_shuffled)

    # Split with fixed ratio
    samples_train = samples_shuffled[0:int(0.8 * len(samples[:, 1])), :]
    samples_eval = samples_shuffled[int(0.8 * len(samples[:, 1])) + 1:len(samples[:, 1]), :]

    if n_clusters > len(samples_train[:, 1]):
        print('Error: número de clusters maior do que o número de vetores de treinamento disponíveis!')
        return None

    # Inicialização de cluster
    if clusters_init is not None:
        clusters = np.copy(clusters_init)
    else:
        init = utils_random.sample_without_replacement(n_population=len(samples_train[:, 1]), n_samples=n_clusters)
        clusters = samples_train[init, :]

    # Allocate
    dist_vec = np.zeros([len(samples_train[:, 1])], dtype='float')
    class_vec = np.zeros([len(samples_train[:, 1])], dtype='int')

    # Mean distance before correction
    dist_init = mean_distance(samples_eval, clusters)

    # Notification
    print('Mean distance gain [dB]:')

    # Correction
    for i in range(max_iterations):
        dist_prev = mean_distance(samples_eval, clusters)
        clusters_prev = clusters

        for j in range(len(samples_train[:, 1])):
            dist_vec[j] = min_distance(samples_train[j, :], clusters)
            class_vec[j] = argmin_distance(samples_train[j, :], clusters)

        for c in range(n_clusters):
            candidate = np.mean(samples_train[class_vec == c], axis=0)

            # Why is this necessary?
            if np.any(np.isnan(candidate)):
                clusters[c, :] = clusters[c, :]
            else:
                clusters[c, :] = candidate

        dist_cur = mean_distance(samples_eval, clusters)
        print(str(i + 1) + ': ' + str(20 * math.log10(dist_cur / dist_init)))

        # Terminacao antes do maximo numero de iteracao
        if dist_cur < dist_thr:
            print('\nTerminate K-means due to achieving distance threshold. Mean distance is {}.\n'.format(dist_cur))
            return clusters, dist_cur

        elif dist_cur / dist_prev > stab_thr:
            print(
                '\nTerminate K-means due to convergence/stability criterion. Mean distance is {}.\n'.format(dist_prev))
            return clusters_prev, dist_prev

    print('\nTerminate K-means due to maximum number of iterations. Mean distance is {}.\n'.format(dist_cur))
    return clusters, dist_cur


# Covariance indicator
def elips(var, cluster, ax, color):
    vals, vecs = np.linalg.eigh(var)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ell = Ellipse(xy=(0, 0), width=2 * np.sqrt(vals[0]), height=2 * np.sqrt(vals[1]),angle=theta, edgecolor=color, fill=False)

    transf = transforms.Affine2D().translate(cluster[0, 0], cluster[0, 1])
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)



### Valor da função da densidade gaussiana para a amostra x, dada a média e variância da distribuição
# x = [1xM], mean = [1xM], var = [MxM]
def normal_dist(x, mean, var):
    # multivariate distribution
    dim = len(x[0, :])

    # normalizacao
    factor = (2 * math.pi) ** (dim / 2) * np.linalg.det(var) ** (1 / 2)
    factor = 1 / factor

    # expoente
    diff_vec = x - mean
    exponent = np.exp(-1 / 2 * np.matmul(np.matmul(diff_vec, np.linalg.inv(var)), np.transpose(diff_vec)))

    # valor escalar
    return np.ndarray.item(factor * exponent)

### Valor do GMM para a amostra x, dados os pesos e as médias e variâncias das gaussianas individuais
# x = [1xM], weights = [1xK], mean = [KxM], var = [KxMxM]
def gmm(x, weights, mean, var):
    prob = 0

    # avalie cada gaussiana
    for i in range(len(weights[0, :])):
        prob = prob + weights[0,i]*normal_dist(x,mean[i,:],var[i,:,:])

    # scalar
    return prob

### Valor da função de custo logarítmico para um determinado conjunto de dados
# samples = [MxN], weights = [1xK], mean = [KxM], var = [KxMxM]
def log_prob(samples, weights, mean, var):
    prob = 0
    for i in range(len(samples[:, 1])):

        prob = prob + math.log(gmm(samples[i,None],weights,mean,var))
        #print(prob)
        #prob = prob + math.log1p(gmm(samples[i, None], weights, mean, var))


    # scalar
    return prob

### Uma iteração do algoritmo EM, incluindo as etapas de Expectativa e Maximização
# samples = [NxM], weights = [1xK], mean = [KxM], var = [KxMxM]
def EM_step(samples, weights, mean, var):
    # Extract parameters and allocate
    n_samples = len(samples[:, 0])
    n_gaussians = len(weights[0, :])
    diff_vec = np.empty((1, len(samples[0, :])))

    # Expectation step
    gamma = np.zeros((n_samples, n_gaussians))
    for i in range(len(samples[:, 1])):

        norm = gmm(samples[i, None], weights, mean, var)

        for j in range(n_gaussians):
            gamma[i, j] = weights[0, j] * normal_dist(samples[i, None], mean[j, None], var[j, None, None]) / norm

    # Maximization step

    # Reiniciar modelo
    weights.fill(0)
    mean.fill(0)
    var.fill(0)

    for k in range(n_gaussians):
        for l in range(n_samples):

            # Update mean and weights
            mean[k, :] = mean[k, :] + gamma[l,k]*samples[l,:]
            weights[0, k] = weights[0, k] + gamma[l,k]

        # Normalization
        norm = weights[0, k]
        weights[0, k] = weights[0, k] / n_samples
        mean[k, :] = mean[k, :] / norm

        for l in range(n_samples):
            # Update variance
            diff_vec[0, :] = np.subtract(samples[l, :], mean[k, :])
            var[k, :, :] = var[k, :, :] + gamma[l, k] * np.matmul(np.transpose(diff_vec), diff_vec)

        # Normalization
        var[k, :, :] = var[k, :, :] / norm


# samples = [NxM], n_gaussians = positive scalar, max_iterations = positive salar
def EM(samples, n_gaussians, max_iterations):
    # Limear da convergencia
    conv_eps = 1 * 10 ** -10

    n_dim = len(samples[0, :])

    # Initialize means
    mean, cost = kmeans_custom(samples, n_gaussians, 20)

    # Initialize variances and weights
    var = np.zeros((n_gaussians, n_dim, n_dim))
    weights = np.zeros((1, n_gaussians))

    # Use codebook prediction results for init
    for i in range(len(samples[:, 1])):
        prediction = argmin_distance(samples[i, :], mean)

        var[prediction, :, :] = var[prediction, :, :] + np.matmul(np.transpose(samples[i,None]-mean[prediction,None]),samples[i,None]-mean[prediction,None])

        weights[0, prediction] = weights[0, prediction] + 1

    # Normalize
    var = var / len(samples[:, 1])
    weights = weights / len(samples[:, 1])

    # Iterate
    for n in range(max_iterations):
        weights_old = np.copy(weights)
        mean_old = np.copy(mean)
        var_old = np.copy(var)

        # Update
        EM_step(samples, weights, mean, var)

        # Abort condition
        if log_prob(samples, weights, mean, var) < log_prob(samples, weights_old, mean_old, var_old) + conv_eps:
            print('EM algorithm has converged.')
            final = [weights_old, mean_old, var_old]
            return final

        # print(log_prob(samples,weights,mean,var))
        # print(log_prob(samples,weights_old,mean_old,var_old))

    print('Exceeding maximum number of EM iterations.\n')
    final = [weights, mean, var]
    return final


### Classification function
# Return index of model with ML
def classify(sample, models):
    n_classes = len(models)
    prob = np.zeros(n_classes)

    for i in range(n_classes):
        prob[i] = gmm(sample, *models[i])

    return np.argmax(prob)


# Return index of model with ML accumulated over multiple samples
def classify_probability(samples, models):
    n_classes = len(models)
    prob = np.zeros(n_classes)

    for i in range(n_classes):
        prob[i] = log_prob(samples, *models[i])
        #print(prob)

    return np.argmax(prob)

