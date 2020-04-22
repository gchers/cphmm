# CPHMM

Python implementation of Conformal Prediction for Hidden Markov Models (CP-HMM) [1].

## Installation

In your chosen virtualenv (otherwise you're recommended to add "--user" as
an argument to setup.py, for a local installation):
```
pip install -r requirements.txt
python setup.py install
```

## Basic usage

cphmm makes use of _nonconformist_ for Conformal Prediction.
Refer to its [docs](https://github.com/donlnz/nonconformist)
for details about nonconformity measures.


```python
import numpy as np
from cphmm.cphmm import CPHMM
from sklearn.neighbors import KNeighborsClassifier
from nonconformist.nc import ClassifierNc, ClassifierAdapter

# X: the observed states (emissions).
# Each row is an observed sequence.
# Emissions can be floats.
# NOTE: it is possible to have sequences of differents
# lengths, but you need to specify `lengths` when
# `fit()`-ing the `CPHMM` below.
X = np.array([[0, 3, 2], [4, 3, 2]])
# H: the corresponding hidden states.
# Each row is an observed sequence.
H = np.array([[0, 2, 1], [0, 0, 2]])

# Define nonconformity measure for CP
knn = KNeighborsClassifier(n_neighbors=3)
ncm = ClassifierNc(ClassifierAdapter(knn))
# Instantiate and train CP-HMM
cphmm = CPHMM(ncm, n_states=3)
cphmm.fit(X, H)

# We predict candidate hidden state sequences for
# a test object.
x_test = np.array([1, 2, 3])
# Significance level (i.e., the probability that the
# correct sequence _will not_ be in the prediction
# set.
significance = 0.2
candidate_sequences = cphmm.predict(x_test, significance)
```

The prediction  `candidate_sequences`
is a list of candidate sequences (tuples).
The probability that the correct hidden sequence (y) is not in *predicted*
is exactly *significance*, thanks to the validity guarantee.

Another simple example is in `examples/example.py`.

If you need help or encounter any issue, please open a ticket.

## Known limitations

Only Transductive Conformal Prediction (CP) is implemented.
Inductive CP may be supported in the future.


## References
[1] "Hidden Markov Models with Confidence" (Giovanni Cherubin and Ilia Nouretdinov, 2016)
