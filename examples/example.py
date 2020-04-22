"""This is a very simplistic example of how to use CPHMM.

This experiment:
  * Assumes a specific HMM.
  * Generates a sample of N examples from that HMM.
  * Uses the first N-1 examples for training, and only makes a prediction on
    the last example.
  * Estimates an HMM on the training set and uses
    the Viterbi algorithm to predict the hidden sequence.
  * Uses CP-HMM to perform the same.
  * Compares the results from Viterbi and CP-HMM.

NOTE: as indicated in (Cherubin&Nouretdinov, 2016), the CPHMM approach will
benefit over the maximum likelihood approach when the samples come from
"unexpected" distributions.
For example, you can try using the standard Viterbi approach assuming a normal
distribution and feed it with data coming from a GMM; because CPHMM guarantees
validity regardless of the distribution, it will outperform Viterbi.
"""
__author__ = 'Giovanni Cherubin (gchers)'

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from nonconformist.nc import ClassifierNc, ClassifierAdapter

from cphmm.cphmm import CPHMM
from experiment_utils import *


if __name__ == "__main__":
    L = 10          # Length of the sample.
    N = 2000        # How many samples.
    H_n = 3         # Number of hidden states.
    SEED = 0
    SIGNIFICANCE_LEVEL = 0.2

    # Initial probabilities, transition probabilities,
    # emission probabilities (means and cov matrices
    # for Gaussian distribution).
    start_prob = np.array([0.6, 0.3, 0.1])
    trans_prob = np.array([[0.7, 0.2, 0.1],
                          [0.3, 0.5, 0.2],
                          [0.3, 0.3, 0.4]])
    emi_means = np.array([0., 2., 4.])
    emi_vars = np.array([1., 1., 1.])

    # Generate samples
    np.random.seed(SEED)
    X, H = sample_hmm(N, L, H_n, start_prob, trans_prob,
                      emi_means, emi_vars)

    # Training and test sets. The test set is only
    # composed by the last sampled sequence.
    train = range(N-1)
    Xtrain = X[train]
    Htrain = H[train]
    Xtest = X[N-1]
    Htest = H[N-1]

    n, l, _ = X.shape
    X = X.flatten()
    H = H.flatten()
    lengths = [l] * n

    # NCM
    knn = KNeighborsClassifier(n_neighbors=1)
    ncm = ClassifierNc(ClassifierAdapter(knn))
    cphmm = CPHMM(ncm, n_states=H_n, smooth=False)

    # HMM trained using Maximum Likelihood.
    ml_pred = ml_hmm_predict(Xtest, Xtrain, Htrain)
    ml_error = error(ml_pred, Htest)
    print("Maximum likelihood error: {}".format(ml_error))

    # CP-HMM training and prediction.
    cphmm.fit(X, H, lengths)
    CP_pred = cphmm.predict(Xtest, SIGNIFICANCE_LEVEL)
    CP_error = error(CP_pred[0], Htest)
    print("CP error: {}".format(CP_error))
