#!/usr/bin/env python
import sys
import itertools
import numpy as np
from nonconformist.cp import TcpClassifier
from nonconformist.nc import BaseScorer

import ml_train


def cp_hmm(xn, X, H, e, ncm, smooth=True, init_prob=None, tran_prob=None):
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
    ncm : nonconformist.BaseScorer
        Nonconformity measure to use.
    smooth : bool
        If True, smooth CP is used, which achieves exact validity.
        Otherwise, standard CP is used, guaranteeing error smaller or
        equal to e.
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
    """
    # Reduce significance level as required.
    e /= float(len(xn))
    # Non conformity measure.
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
        p = map(int, p)         # So we can use them as indexes
        s = init_prob[p[0]]
        for i in range(len(p)-1):
            s *= trans_prob[(p[i],p[i+1])]
        scores.append(s)
    
    paths_sorted = [x[1] for x in sorted(zip(scores, paths), reverse=True)]

    return paths_sorted

def cp_hmm_candidates(xn, X, H, e, ncm, smooth=True):
    """Uses CP-HMM to predict, for each element of the observed sequence, a
    list of candidate states.  Thanks to CP's validity guarantee, the true
    hidden states is within the list of candidates with probability 1-e.
    Accepts training examples X and H, where X is a list of observed sequences,
    H is a list of respective hidden sequences.  e is the significance level,
    and represents the error we can afford to make.

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
    ncm : nonconformist.BaseScorer
        Nonconformity measure to use.
    smooth : bool
        If True, smooth CP is used, which achieves exact validity.
        Otherwise, standard CP is used, guaranteeing error smaller or
        equal to e.
    """
    # Flatten sequences
    X = X.flatten().reshape(-1, 1)
    H = H.flatten()
    # For each element of the observed sequence xn
    # determine a set of candidate states.
    cp = TcpClassifier(ncm, smoothing=smooth)
    hn_candidates = []
    for x in xn:
        cp.fit(X, H)
        candidates_bool = cp.predict(x.reshape(-1, 1), e)[0]
        candidates = cp.classes[candidates_bool]
        hn_candidates.append(candidates)

    return hn_candidates
