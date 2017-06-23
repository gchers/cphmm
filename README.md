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
Refer to [https://github.com/donlnz/nonconformist](the original manual)
for details about nonconformity measures.


```
from cphmm import CPHMM
from sklearn.neighbors import KNeighborsClassifier
from nonconformist.nc import ClassifierNc, ClassifierAdapter

X = []
Y = []
lengths = []
x = []
y = []
significance = 0.2

# Define a k-NN nonconformity measure
knn = KNeighborsClassifier(n_neighbors=3)
ncm = ClassifierNc(ClassifierAdapter(knn))
# Instantiate and train CP-HMM
cphmm = CPHMM(ncm)
cphmm.fit(X, Y, lengths)
# Predict
predicted = cphmm.predict(x, significance)
```

The prediction is a list of candidate sequences (tuples).
The probability that the correct hidden sequence (y) is not in *predicted*
is exactly *significance*, thanks to the validity guarantee.

## Known limitations

Only Transductive Conformal Prediction (CP) is implemented.
Inductive CP may be supported in the future.


## References
[1] "Hidden Markov Models with Confidence" (Giovanni Cherubin and Ilia Nouretdinov, 2016)
