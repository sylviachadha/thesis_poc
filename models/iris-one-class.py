# IRIS DATASET AS ONE CLASS CLASSIFICATION PROBLEM
# ----------------------------------------------------
# TRAIN ONLY ON CLASS 0 BUT PUT CLASS 1 IN TEST

# STEP 1 Import sklearn and load dataset
# -------------------------------------------
from sklearn.datasets import load_iris
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

iris_bunch = load_iris()
# The data structure is bunch and loads data, feature_names, target, target_names etc

# To see X or features iris_bunch.data as ndarray
print(iris_bunch.data)
print(iris_bunch.data.shape)
print("feature_names", iris_bunch.feature_names)
X_iris = iris_bunch.data

# To see corresponding labels / target Y
print(iris_bunch.target)
print(iris_bunch.target.shape)
print("target_names", iris_bunch.target_names)
y_iris = iris_bunch.target

# Summarize class distribution
counter = Counter(iris_bunch.target)
print(counter)

# STEP 2 - Data Pre-Preprocessing, change to Binary Problem
# ------------------------------------------------------------
# Create new labels and corresponding data with only 2 labels

X = iris_bunch.data
y = iris_bunch.target

# Only keep that data corresponding to which label is 0 and 1
X1 = X[y == 0]
X2 = X[y == 1]
X_Binary = np.concatenate((X1, X2))
print("X length", len(X_Binary))

y_Binary = y[y != 2]
print("y length", len(y_Binary))

# Summarize class distribution
counter1 = Counter(y_Binary)
print(counter1)

# STEP 3 - Data Pre-Preprocessing, change to Imbalanced classes
# ---------------------------------------------------------------

# Step 3A - Reduce class=1/versicolor 50 samples to just 3 samples
# by removing 47 observations from tail of X - two dimensional array
n = 47
X_short = X_Binary[:-n, :]
print("Filtered data", X_short)
print("Length of Filtered Data", len(X_short))

# Remove 47 observations from tail of Y
y_short = np.delete(y_Binary, range(53, 100, 1))

# STEP 4 - Split into Train & Test to feed to ML Algorithm
# ----------------------------------------------------------
# Check Shape of data and target/label
print("Shape of data", X_short.shape)
print("Shape of label", y_short.shape)

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_short, y_short, test_size=0.2, random_state=4)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

# Step 5 Modelling  A. Import B. Instantiate C. Fit  D. Predict
# --------------------------------------------------------------

# Model 1 # One Class SVM
# --------------------------------------------------------------

# A. Import
from sklearn.svm import OneClassSVM

# B. Instantiate
ocsvm = OneClassSVM(gamma='scale', nu=0.05)

# C. Train
X_train = X_train[y_train == 0]
ocsvm.fit(X_train)

# D. Predict
ypred1 = ocsvm.predict(X_test)
print("y_pred", ypred1)

# Model 2 # Isolation Forest
# --------------------------------------------------------------

# A. Import
from sklearn.ensemble import IsolationForest

# B. Instantiate
iforest = IsolationForest(contamination=0.05)

# C. Train on Majority class only
#X_train = X_train[y_train == 0] # Already filtered X_train in ocsvm
# so just fit
iforest.fit(X_train)

# D. Predict
ypred2 = iforest.predict(X_test)
print("y_pred", ypred2)


# Model 3 # Minimum Covariance Determinant
# -----------------------------------------------------------
# Useful when input has gaussian distribution

# A. Import
from sklearn.covariance import EllipticEnvelope

# B. Instantiate
e_envelope = EllipticEnvelope(contamination=0.05)

# C. Train on Majority class only
#X_train = X_train[y_train == 0] # Already filtered X_train in ocsvm
# so just fit
e_envelope.fit(X_train)

# D. Predict
ypred3 = e_envelope.predict(X_test)
print("y_pred", ypred3)


# Step 6 - Change actual y_test to 1 & -1 as y_pred is returned in this form
# Mark inliers 1, outliers -1 in test_y
# -----------------------------------------------------------------------------

y_test[y_test == 1] = -1
y_test[y_test == 0] = 1
print("y_test", y_test)

# Step 7 - Evaluation Metrics
# -----------------------------------------

# Define generalized function

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return(cm, acc, recall, precision, f1score)

# Model 1 Logistic Regression
ocsvm_result = evaluate_model(y_test, ypred1)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result)

# Model 2 Isolationn Forest
iforest_result = evaluate_model(y_test, ypred2)
print("Isolation Forest values for cm, accuracy, recall, precision and f1 score", iforest_result)

# Model 3 Minimum Covariance Determinant
eenvelope_result = evaluate_model(y_test, ypred3)
print("MCD values for cm, accuracy, recall, precision and f1 score", eenvelope_result)



# STEP 8 - Make another test set with more anomaly (class versicolor)
# Add 40 egs of anomaly class
# --------------------------------------------------------------------
# Step 8A Make y
y_Binary1 = y_Binary.copy()
y_test_xl = y_test.copy()

y_test_xl1 = np.delete(y_Binary1, range(0, 60, 1))
# Replace 1 with -1 for anomaly class
y_test_xl1[y_test_xl1 == 1] = -1

y_test_merged = np.concatenate((y_test_xl, y_test_xl1))
print("Merged labels of new test set", y_test_merged)

# Step 8B Make X

X_test_xl = X_test.copy()
X_Binary1 = X_Binary.copy()

# Keep 40 observations from tail of X
n = 40
X_Binary1 = X_Binary1[-n:, :]
print("Filtered data", X_Binary1)
print("Length of Filtered Data", len(X_Binary1))

# Merge X_test_xl and new_data_X1_copy
X_test_merged = np.concatenate((X_test_xl, X_Binary1))
print("Length of new test set", len(X_test_merged))


# STEP 9 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset
# Model 1 One class SVM
ypred4 = ocsvm.predict(X_test_merged)  # Predict
ocsvm_result1 = evaluate_model(y_test_merged, ypred4)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result1)

# Model 2 Isolation Forest
ypred5 = iforest.predict(X_test_merged)  # Predict
iforest_result1 = evaluate_model(y_test_merged, ypred5)
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result1)

# Model 3 MCD
ypred6 = e_envelope.predict(X_test_merged)  # Predict
eenvelope_result1 = evaluate_model(y_test_merged, ypred6)
print("MCD values for cm, accuracy, recall, precision and f1 score", eenvelope_result1)

