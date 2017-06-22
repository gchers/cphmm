#!/usr/bin/env python
import sys
import itertools
import numpy as np
from sklearn.base import BaseEstimator
from nonconformist.nc import BaseScorer
from nonconformist.cp import TcpClassifier

import ml_train


class CPHMM(BaseEstimator):


    def __init__(self, ncm, smooth=True):
        """Initialise a CP-HMM model.

        Parameters
        ----------
        ncm : nonconformist.BaseScorer
            Nonconformity measure to use.
        smooth : bool
            If True, smooth CP is used, which achieves exact validity.
            Otherwise, standard CP is used, guaranteeing error smaller or
            equal to the significance level.
        """
        self.ncm = ncm
        self.smooth = smooth

    def fit(self, X, Y, init_prob=None, tran_prob=None):
        """Fits the model on observables X and respective hidden sequences Y.

        Parameters
        ----------
        X : numpy array
            Two dimensional array. Each row is an observed sequence.
        Y : numpy array
            Two dimensional array. Each row is a hidden sequence.
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
        self.train_x = X
        self.train_y = Y

        # CP model
        self.cp = TcpClassifier(self.ncm, smoothing=self.smooth)

        # Initial and transition probabilities.
        if not init_prob:
            ip = ml_train.estimate_initial_prob(Y)
            init_prob = dict(zip(range(len(ip)), ip))

        if not tran_prob:
            tp = ml_train.estimate_transition_prob(Y)
            # Convert transmission probabilities into dictionary.
            tran_prob = {}
            for i in range(len(tp)):
                for j in range(len(tp[0])):
                    tran_prob[(i,j)] = tp[i][j]
    
        self.init_prob = init_prob
        self.tran_prob = tran_prob

    def predict(self, x, e):
        """Return a CP-HMM prediction region.

        Uses CP-HMM to output a prediction region (i.e.: a set of candidate
        hidden sequences) for the observed sequence x.
        NOTE: If any of the elements of the sequence has an empty
        prediction set, then no predictions are returned.

        Parameters
        ----------
        x : list
            Observed sequence.
        e : float in [0,1]
            Significance level.
        """
        # Reduce significance level as required.
        e /= float(len(x))
        # Find candidates.
        candidates = self._hidden_candidates(x, e)

        # Generate paths.
        paths = self._generate_paths(candidates, self.tran_prob,
                                     self.init_prob)

        return paths

    def _generate_paths(self, candidates, trans_prob, init_prob):
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

    def _hidden_candidates(self, x, e):
        """Uses CP-HMM to predict, for each element of the observed sequence, a
        list of candidate states.  Thanks to CP's validity guarantee, the true
        hidden states is within the list of candidates with probability 1-e.

        Parameters
        ----------
        x : list
            Observed sequence.
        e : float in [0,1]
            Significance level.
        """
        # Flatten sequences, "train" CP
        X = self.train_x.flatten().reshape(-1, 1)
        Y = self.train_y.flatten()
        self.cp.fit(X, Y)
        # For each element of the observed sequence x
        # determine a set of candidate states.
        y_candidates = []
        for i in range(len(x)):
            candidates_bool = self.cp.predict(x[i].reshape(-1, 1), e)[0]
            candidates = self.cp.classes[candidates_bool]
            y_candidates.append(candidates)

        return y_candidates
