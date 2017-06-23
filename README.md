# CPHMM

Python implementation of Conformal Prediction for Hidden Markov Models (CP-HMM) [1].

## Installation

## Basic usage

```
from cphmm import CPHMM
from sklearn.neighbors import KNeighborsClassifier
from nonconformist.nc import ClassifierNc, ClassifierAdapter

X = []
Y = []
lengths = []
x = []
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



## References
[1] "Hidden Markov Models with Confidence" (Giovanni Cherubin and Ilia Nouretdinov, 2016)
