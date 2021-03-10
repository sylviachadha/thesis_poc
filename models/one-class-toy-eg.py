# One Class Toy Example (With Synthetic Data)
# ------------------------------------------------------------------
# Generate and plot a synthetic imbalanced classification dataset

# Step 1 Import libraries
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Step 2 - Define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)

# Summarize class distribution
counter = Counter(y)
print(counter)

# Step 3 - Scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

# Step 4 - Split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

# Step 5 - Modelling # A. Import B. Instantiate C. Fit D. Predict

# Step A Import
from sklearn.svm import OneClassSVM

# Step B Instantiate
ocsvm = OneClassSVM(gamma='scale', nu=0.01)

# Step C Fit / Train
# Below code only includes in train the majority class i.e
# all rows of train_X which has corresponding train_y = 0
# There are 4995 such rows out of 5000
trainX = trainX[trainy == 0]
# We just fit only on trainX, no label or trainy is provided
# hence this is unsupervised learning where normal class
# features(X) are learnt and abnormal is detected based on deviations
# from the normal.

# One Class because you fit only on the majority class not on other class
# and unsupervised because u do not provide labels
ocsvm.fit(trainX)

# AE is semi-supervised because u provide the labels but they are
# same as the features i.e. the idea is to reconstruct model.fit(trainX,trainX)

# Step D Predict
# The model is built in such a way it does not need labels, it just learns
# the features of majority class. Gives 1 for inliers(majority class) and
# -1 for outliers(which it detects have different features from the learnt
# features of majority class)

yhat = ocsvm.predict(testX)

# Step 6 - Change actual y_test to 1 & -1 as y_pred is returned in this form
# Mark inliers 1, outliers -1 in test_y

testy[testy == 1] = -1
testy[testy == 0] = 1

# Step 7 - Calculate f1score
score = f1_score(testy, yhat, pos_label=-1)
print('F1 Score: %.3f' % score)
