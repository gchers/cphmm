"""Utilities.
"""
__author__ = 'Giovanni Cherubin (gchers)'

import numpy as np
import sys
from hmmlearn import hmm
from sklearn import mixture

from ml_train import estimate_initial_prob, estimate_transition_prob, \
                       estimate_emission_gauss

# DEFAULT PARAMETERS

# 3 states, Normal distribution.
params_hmm_norm_0 = {'n_components': 3,
                     'start_prob': np.array([0.6, 0.3, 0.1]),
                     'trans_prob': np.array([[0.7, 0.2, 0.1],
                                             [0.3, 0.5, 0.2],
                                             [0.3, 0.3, 0.4]]),
                     'emi_means': np.array([-2, 0., 2]),
                     'emi_vars': np.array([0.5, 0.5, 0.5])
                    }

# 3 states, GMM distribution.
params_hmm_gmm_0 = {'n_components': 3,
                    'n_mix': 2,
                    'start_prob': np.array([0.6, 0.3, 0.1]),
                    'trans_prob': np.array([[0.7, 0.2, 0.1],
                                            [0.3, 0.5, 0.2],
                                            [0.3, 0.3, 0.4]]),
                    'emi_means': np.array([[0., 2.],
                                           [-2., -1.],
                                           [2., 3.]]),
                    'emi_vars': np.array([[0.49, 0.49],
                                          [0.0625, 0.0625],
                                          [0.25, 0.09]]),
                    'emi_weights': np.array([[0.7, 0.3],
                                             [0.5, 0.5],
                                             [0.7, 0.3]])
                   }





def sample_hmm(N, L, n_components, start_prob, trans_prob,
               emi_means, emi_vars):
    """Sample N observations of size L from an HMM.

    Samples N observations of size L from an HMM with the defined
    properties. Returns N observations and the corresponding
    N hidden states. Emission probability is modelled by
    a gaussian pdf. Observed values are continuous scalars
    (NOT vectors).

    Parameters
    ----------
    N : int
        Number of observations to sample.
    L : int
        Elements for each observation.
    n_components : int
        Number of hidden states.
    start_prob : numpy array
        One dimensional array containing n_components floats.
        The sum of the elements should be 1.
    trans_prob : numpy array
        Two dimensional array with dimension
        (n_components, n_components). Each row and col rappresent
        a state, and each element e_ij (float in [0,1])
        is the probability of going from state i to state j.
        The sum of each row is 1.
    emi_means : numpy array
        One dimensional array containing n_components floats.
        The i-th element is the mean of a Gaussian distribution
        that models the emission from the i-th state.
    emi_vars : numpy array
        One dimensional array containing n_components floats.
        The i-th element is the variance of a Gaussian distribution
        that models the emission from the i-th state.
    """
    # Define true model.
    true_model = hmm.GaussianHMM(n_components=3,
                            covariance_type="diag")

    true_model.startprob_ = start_prob
    true_model.transmat_ = trans_prob
    means = np.array([[x] for x in emi_means])
    covars = np.array([[x] for x in emi_vars])
    true_model.means_ = means
    true_model.covars_ = covars

    # Generate samples.
    X = []              # Observations.
    H = []              # Hidden sequences to predict.
    for i in range(N):
        x, h = true_model.sample(L)
        X.append(x)
        H.append(h)
    X = np.array(X)
    H = np.array(H)

    return X, H

def sample_hmm_gmm(N, L, n_components, n_mix, start_prob, trans_prob,
               emi_means, emi_vars, emi_weights):
    """Sample N observations of size L from an HMM.

    Samples N observations of size L from an HMM with the defined
    properties. Returns N observations and the corresponding
    N hidden states. Emission probability is modelled by
    a gaussian pdf. Observed values are continuous scalars
    (NOT vectors).

    Parameters
    ----------
    N : int
        Number of observations to sample.
    L : int
        Elements for each observation.
    n_components : int
        Number of hidden states.
    n_mix : int
        Number of mixtures for the GMM.
    start_prob : numpy array
        One dimensional array containing n_components floats.
        The sum of the elements should be 1.
    trans_prob : numpy array
        Two dimensional array with dimension
        (n_components, n_components). Each row and col rappresent
        a state, and each element e_ij (float in [0,1])
        is the probability of going from state i to state j.
        The sum of each row is 1.
    emi_means : numpy array with shape (n_components, n_mix)
        Each row contains the means for the components of
        the GMM.
    emi_vars : numpy array with shape (n_components, n_mix)
        Each row contains the variances for the components
        of the GMM.
    emi_weights: numpy array with shape (n_components, n_mix)
        Each row contains the weights for the components
        of the GMM.
        Rows should sum up to 1.
    """
    # Define true model.
    gmmhmm = hmm.GMMHMM(n_components=n_components, n_mix=n_mix,
                        startprob_prior=1.0, algorithm="viterbi",
                        covariance_type='diag')

    gmmhmm.startprob_ = start_prob
    gmmhmm.transmat_ = trans_prob

    gmms = []
    for i in range(n_components):
        gmm = mixture.GaussianMixture(n_components=n_mix,
                                      covariance_type='diag')
        gmm.means_ = np.array([[x] for x in emi_means[i,:]])
        gmm.covars_ = np.array([[x] for x in emi_vars[i,:]])
        gmm.weights_ = emi_weights[i,:]
        gmms.append(gmm)

    gmmhmm.startprob_ = start_prob
    gmmhmm.transmat_ = trans_prob
    gmmhmm.gmms_ = gmms

    # Generate samples.
    X = []              # Observations.
    H = []              # Hidden sequences to predict.
    for i in range(N):
        x, h = gmmhmm.sample(L)
        X.append(x)
        H.append(h)
    X = np.array(X)
    H = np.array(H)

    return X, H

def ml_hmm_predict(xtest, Xtrain, Htrain):
    """Train HMM and predict the most likely states sequence.

    Trains a HMM having fully observable data using
    the Maximum Likelihood method. Predicts the hidden
    states corresponding to the observed sequence xtest
    using the Viterbi algorithm.

    Parameters
    ----------
    xtest : numpy array
        One dimensional array containing the observed sequence.
    Xtrain : numpy array
        Two dimensional array containing training observed
        sequences.
    Htrain : numpy array
        Two dimensional array containing training states
        sequences corresponding to Xtrain.
    """
    # HMM trained using ML.
    ip = estimate_initial_prob(Htrain)
    A = estimate_transition_prob(Htrain)
    m, v = estimate_emission_gauss(Xtrain, Htrain)

    ml_model = hmm.GaussianHMM(n_components=3,
                                 covariance_type="diag")
    ml_model.startprob_ = ip
    ml_model.transmat_ = A
    ml_model.means_ = m
    ml_model.covars_ = v

    return list(ml_model.predict(xtest))

def error(s1, s2):
    """Returns an error score, defined as the
    number of elements in which the two sequences
    differ (Hamming distance) normalised in [0,1].
    """
    if len(s1) != len(s2):
        return -1

    score = sum([1. if s1[i]!=s2[i] else 0.
                 for i in range(len(s1))])

    return float(score)/len(s1)
