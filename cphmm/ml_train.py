#!/usr/bin/env python
import numpy as np
import sys
from hmmlearn import hmm

def estimate_initial_prob(H):
    """Accepts a list of sequences. Each sequence
    is a list of floats.
    Returns an frequentist estimate of the initial
    probabilities over the observed hidden states.
    Assumes that the hidden states are specified by
    sequential numbers starting from 0 (with no
    missing numbers).
    """
    if len(H) == 0:
        return []
    
    H_n = len(np.unique(H))
    ip = np.array([0.0]*H_n)

    for h in H:
        ip[h[0]] += 1

    return ip/len(H)

def estimate_transition_prob(H):
    """Accepts a list of sequences. Each sequence
    is a list of ints.
    Returns an frequentist estimate of the transition
    probabilities over the observed hidden states.
    Assumes that the hidden states are specified by
    sequential numbers starting from 0 (with no
    missing numbers).
    H_n is the number of states.
    The return value is a numpy matrix in the form:
        
        P(h_1 -> h_1), P(h_2 -> h_1), ... 
        P(h_2 -> h_1), p(h_2 -> h_2), ...
        ...

    where P(h_i -> h_j) indicates the transition
    probability from state h_i to state h_j.
    """
    if len(H) == 0:
        return []
    
    H_n = len(np.unique(H))
    tp = np.zeros((H_n, H_n))

    for h in H:
        for i in range(H_n-1):
            tp[h[i],h[i+1]] += 1

    for i in range(H_n):
        if sum(tp[i,:]) == 0:
            tp[i,:] = 1.0
        tp[i,:] /= sum(tp[i,:])

    return tp
    
    
def estimate_emission_gauss(X, H):
    """Accepts a list of observed sequences X and
    of respective hidden sequences H. Each sequence
    is a list of ints.
    We assume that P(x_i | h_i) (emission probability)
    is normally distributed ~N(m_i, s_i).
    The function returns the array of means m_i
    and variances s_i estimated from data.
    """
    if len(H) == 0 or len(H) != len(X):
        return []
    
    H_n = len(np.unique(H))
    g_mean = [[]]*H_n
    #g_var = np.tile(np.identity(1)*0, (H_n, 1, 1))
    g_var = [[]]*H_n

    for i in range(H_n):
        idx_i = np.where(H==i)
        g_mean[i] = [np.mean(X[idx_i])]
        g_var[i] = [np.var(X[idx_i])]

    return (np.array(g_mean), np.array(g_var))


def estimate_emission_prob(X, H, X_n):
    """Accepts a list of observed sequences X and
    of respective hidden sequences H. Each sequence
    is a list of ints.
    Returns an frequentist estimate of the emission
    probabilities from the observed hidden states
    to the observations.
    Assumes that the hidden states are specified by
    sequential numbers starting from 0 (with no
    missing numbers).
    H_n is the number of states.
    The return value is a numpy matrix in the form:
        
        P(x_1 | h_1), P(x_1 | h_2), ... 
        P(x_1 | h_2), p(x_2 | h_2), ...
        ...
    where P(x_i | h_j) is the probability of observing
    x_i given the hidden state is h_j.
    """
    if len(H) == 0 or len(H) != len(X):
        return []
    
    H_n = len(np.unique(H))
    ep = np.zeros((H_n, X_n))

    for i in range(H_n):
        for j in range(X_n):
            ep[H[i,j], X[i,j]] += 1

    for i in range(H_n):
        ep[i,:] /= sum(ep[i,:])

    return ep
