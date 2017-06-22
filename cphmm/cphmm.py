#!/usr/bin/env python
import sys
import itertools
import numpy as np
from cpy.cp import CP
from cpy.nonconformity_measures import knn

import ml_train


def cp_hmm(xn, X, H, e, init_prob=None, tran_prob=None, ncm='default', smooth=True):
    """CP-HMM prediction region.

    Uses CP-HMM to output a prediction region (i.e.: a set
    of candidate hidden sequences) for the observed sequence
    xn.

    Parameters
    ----------
    xn : list
        Observed sequence.
    X : numpy array
        Two dimensional array. Each row is an observed sequence.
    H : numpy array
        Two dimensional array. Each row is a hidden sequence.
    e : float in [0,1]
        Significance level.
    init_prob : dict (Default: None)
        The item corresponding to the i-th key is the
        probability of the hidden process to start in the
        i-th state.
        If default (=None), it is estimated from data.
    tran_prob : dict (Default: None)
        The item corresponding to the i-th key of the dictionary
        is a dictionary itself, which, for the j-th key,
        indicates the probability of transitioning from the
        i-th to the j-th state.
        If default (=None), it is estimated from data.
    ncm : string
        Non conformity measure. The default is k-NN non conformity
        measure with k=1.
    """
    # Reduce significance level as required.
    e /= float(len(xn))
    # Non conformity measure.
    if ncm == 'default':
        ncm = knn.KNN(k=1)
    # Convert sequences to training set.
    # Find candidates.
    candidates = cp_hmm_candidates(xn, X, H, e, ncm, smooth)
    # Initial and transition probabilities.
    if not tran_prob:
        tp = ml_train.estimate_transition_prob(H)
        # Convert transmission probabilities into dictionary.
        #tp_d = {(i,j): tp[i][j] for i in range(len(tp)) for j in range(len(tp[0]))}
        tp_d = {}
        for i in range(len(tp)):
            for j in range(len(tp[0])):
                tp_d[(i,j)] = tp[i][j]
    else:
        tp_d = tran_prob

    if not init_prob:
        ip = ml_train.estimate_initial_prob(H)
        ip_d = dict(zip(range(len(ip)), ip))
    else:
        ip_d = init_prob

    # Generate paths.
    # NOTE: If any of the elements of the sequence has an empty
    # prediction set, then no paths are returned.
    paths = gen_paths(candidates, tp_d, ip_d)

    return paths

def gen_paths(candidates, trans_prob, init_prob):
    """Generate and score paths.

    Accepts a list of list of candidate. Each list of
    candidate contains potential true hidden states to
    compose a path.
    The function produces all possible paths, and scores them
    w.r.t. the transition and initial probabilities.
    It returns the paths in a list, sorted by scores:
    from the most likely to the least likely.

    Parameters
    ----------
    candidates : list of list
        The i-th list it contains represents a set of state
        candidates for the i-th element of the sequence.
    trans_prob : dictionary
        The keys of this dictionary are tuples in the form
        (i, j). The element (i, j) is associated with the
        transition  probability from state i to state j.
    init_prob : dictionary
        The keys are numbers i (as many as the states).
        init_prob[i] contains the probability of a sequence
        starting in state i.
    """
    paths = list(itertools.product(*candidates))
    scores = []
    for p in paths:
        s = init_prob[p[0]]
        for i in range(len(p)-1):
            s *= trans_prob[(p[i],p[i+1])]
        scores.append(s)
    
    paths_sorted = [x[1] for x in sorted(zip(scores, paths), reverse=True)]

    return paths_sorted

#def cp_hmm_priors(H, e, ncm, **ncm_args):
#    """Produces the priors for the CP-HMM.
#
#    Parameters
#    ----------
#    H : numpy array
#        Two dimensional array. Each row is an hidden sequence.
#    e : float
#        Significance level in [0,1].
#    """
#    ns = len(np.unique(H))
#    Htrain = np.array([[h[0]] for h in H])
#    priors = np.array([.0]*ns)
#    for i in range(ns):
#        if CP.predict_unlabelled(i, Htrain, e, ncm, **ncm_args):
#            priors[i] = 1.
#    
#    return priors
#
#def cp_hmm_transition_matrix(H, e, ncm, **ncm_args):
#    """Produces the transition matrix for the CP-HMM.
#
#    Parameters
#    ----------
#    H : numpy array
#        Two dimensional array. Each row is an hidden sequence.
#    e : float
#        Significance level in [0,1].
#    """
#    Ht1 = []            # H at step "t1"
#    Ht2 = []            # H at step "t2"
#    for h in H:
#        for i in range(len(h)-1):
#            Ht1.append([h[i]])
#            Ht2.append(h[i+1])
#    states = np.unique(H)
#    N = len(states)
#    tran_matrix = np.zeros((N, N))
#    for i in range(N):
#        # Predict which states follow states[i].
#        follow = CP.predict_labelled(states[i], Ht1, Ht2, e, ncm, **ncm_args)
#        tran_matrix[i, follow] = 1.
#
#    return tran_matrix


def sequences_to_examples(X, H):
    """Transforms observed/hidden sequences to a training set
    of examples.

    This function accepts a list of observed sequences and
    a list of the respective hidden sequences.
    It returns two lists, Xt, Ht; the first one contains
    observed elements, and the second one contains the
    corresponding hidden states.

    Parameters
    ----------
    X : numpy array
        Array of sequences (arrays).
    H : numpy array
        Two dimensional array. Each row is an hidden sequence.
    """
    Xt = []
    Ht = []
    for i in range(len(X)):
        Xt.append(X[i])
        Ht.append(H[i])

    return Xt, Ht

def cp_hmm_candidates(xn, X, H, e, ncm, smooth=True):
    """Uses CP-HMM to predict, for each element of the observed
    sequence, a list of candidate states.
    Thanks to CP's validity guarantee, the true hidden
    states is within the list of candidates with probability
    1-e.
    Accepts training examples X and H, where X is a list of
    observed sequences, H is a list of respective hidden
    sequences.
    e is the significance level, and represents the error we
    can afford to make.
    """
    Xt, Ht = sequences_to_examples(X, H)
    # For each element of the observed sequence xn
    # determine a set of candidate states.
    cp = CP(ncm, smooth)
    hn_candidates = []
    for x in xn:
        candidates = cp.predict_labelled(x, Xt, Ht, e)
        hn_candidates.append(candidates)

    return hn_candidates

def cp_hmm_pvalues(xn, hn, X, H, ncm, smooth=True):
    """Returns the p-values for the observed sequence given
    its correct hidden sequence.

    """
    cp = CP(ncm, smooth)
    Xt, Ht = sequences_to_examples(X, H)
    Xt = np.array(Xt)
    Ht = np.array(Ht)
    pvalues = []
    for j in range(len(xn)):
        p = cp.calculate_pvalue(xn[j], Xt[Ht==hn[j],:], ncm)
        pvalues.append(p)

    return pvalues
