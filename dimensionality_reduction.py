import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]


from sklearn.ensemble import RandomForestClassifier
rand_forest_classifier = RandomForestClassifier(random_state=42)

import time

t0 = time.time()
rand_forest_classifier.fit(X_train, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1-t0))

from sklearn.metrics import accuracy_score
y_pred = rand_forest_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # automatically select number of PC to get 95% variance
X_train_reduced = pca.fit_transform(X_train)

rand_forest_classifier2 = RandomForestClassifier(random_state=42)
t0 = time.time()
rand_forest_classifier2.fit(X_train_reduced, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1-t0))  # for some reason, the PCA version takes longer time to train!

X_test_reduced = pca.transform(X_test)
y_pred = rand_forest_classifier2.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))

