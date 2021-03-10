# IRIS DATASET AS BINARY CLASSIFICATION PROBLEM
# EVALUATE DIFFERENT ALGORITHMS ON IMBALANCED BINARY LABELLED DATASET
# ---------------------------------------------------------------------
# CLASS PERCENTAGES 94.3 % MAJORITY AND 5.7 % MINORITY CLASS
# ---------------------------------------------------------------------
# PART A - TOTAL 53 DATA = 42 TRAIN DATA AND 11 TEST DATA
# ---------------------------------------------------------------------
# 11 TEST DATA has just 1 anomaly or observation of minority class
# ---------------------------------------------------------------------


# STEP 1 Import sklearn and load dataset
# -------------------------------------------
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np

iris_bunch = load_iris()
# The data structure is bunch and loads data, feature_names, target, target_names etc

# To see X or features iris_bunch.data as ndarray
print(iris_bunch.data)
print(iris_bunch.data.shape)
print("feature_names", iris_bunch.feature_names)

# To see corresponding labels / target Y
print(iris_bunch.target)
print(iris_bunch.target.shape)
print("target_names", iris_bunch.target_names)

# summarize class distribution
counter = Counter(iris_bunch.target)
print(counter)

# STEP 2 - Data Pre-Preprocessing, change to Binary Problem
# ------------------------------------------------------------
# Create new labels and corresponding data with only 2 labels

# Delete all occurrences of 2 from the target numpy array
new_target_y = iris_bunch.target[iris_bunch.target != 2]
print("Filtered Target", new_target_y)
print("Length of Filtered Target", len(new_target_y))

# Also need to delete corresponding X values of virginica(2) label
# which is the last 50 rows of data

n = 50
new_data_X = iris_bunch.data[:-n, :]
print("Filtered data", new_data_X)
print("Length of Filtered Data", len(new_data_X))

# Check Target unique values & counts
import numpy as np

# # summarize class distribution
counter = Counter(new_target_y)
print(counter)

# STEP 3 - Data Pre-Preprocessing, change to Imbalanced classes
# ---------------------------------------------------------------
# setosa(0) = 95 samples and versicolor(1) = 5 samples

# Step 3A - Reduce versicolor 50 samples to just 5 samples
print(new_data_X)
print(new_target_y)

# Remove 47 observations from tail of X
n = 47
new_data_X1 = new_data_X[:-n, :]
print("Filtered data", new_data_X1)
print("Length of Filtered Data", len(new_data_X1))

# Remove 47 observations from tail of Y
import pandas as pd

new_target_y1_df = pd.DataFrame(new_target_y)
new_target_y1_df.drop(new_target_y1_df.tail(n).index,
                      inplace=True)
# Changed back df again to numpy 1D array and flattened
new_target_y1_df = new_target_y1_df.to_numpy().flatten()

# STEP 3 - Split into Train & Test to feed to ML Algorithm
# ----------------------------------------------------------
# Check Shape of data and target/label
print("Shape of data", new_data_X1.shape)
print("Shape of label", new_target_y1_df.shape)

# Split into Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_data_X1, new_target_y1_df, test_size=0.2, random_state=4)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

# STEP 4 - Modelling (1.Import, 2.Instantiate, 3.Fit and 4.Predict)
# ------------------------------------------------------------------

# Model 1 - Logistic Regression
# ------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression  # Import

logreg = LogisticRegression(max_iter=200)  # Instantiate
logreg.fit(X_train, y_train)  # Fit
y_pred_lr = logreg.predict(X_test)  # Predict
print("actual", y_test)
print("LR predict", y_pred_lr)

# Model 2 - KNN
# ------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier  # Import

knn = KNeighborsClassifier(n_neighbors=5)  # Instantiate
knn.fit(X_train, y_train)  # Fit
y_pred_knn = knn.predict(X_test)  # Predict
print("actual", y_test)
print("knn predict", y_pred_lr)

# Model 3 - Decision Tree
# ------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier  # Import

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)  # Instantiate
dt.fit(X_train, y_train)  # Fit
y_pred_dt = dt.predict(X_test)  # Predict
print("actual", y_test)
print("dt predict", y_pred_dt)

# Model 4 - Random Forest
# ------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier  # Import

rf = RandomForestClassifier(n_estimators=200)  # Instantiate
rf.fit(X_train, y_train)  # Fit
y_pred_rf = rf.predict(X_test)  # Predict
print("actual", y_test)
print("rf predict", y_pred_rf)

# Model 5 - SVM
# ------------------------------------------------------------------

from sklearn.svm import SVC  # Import

svc = SVC()  # Instantiate

svc.fit(X_train, y_train)  # Fit

y_pred_svc = svc.predict(X_test)  # Predict
print("actual", y_test)
print("rf predict", y_pred_svc)

# Model 6 - Kernel SVM
# ------------------------------------------------------------------

from sklearn.svm import SVC  # Import

svc_rbf = SVC(kernel='rbf', gamma=1.0)  # Instantiate

svc_rbf.fit(X_train, y_train)  # Fit

y_pred_svcrbf = svc_rbf.predict(X_test)  # Predict
print("actual", y_test)
print("rf predict", y_pred_svcrbf)

# Model 7 - Naive Bayes
# ------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB  # Import

nbayes = GaussianNB()  # Instantiate

nbayes.fit(X_train, y_train)  # Fit (features, labels)

y_pred_nb = nbayes.predict(X_test)  # Predict
print("actual", y_test)
print("nb predict", y_pred_nb)


# STEP 5 - Evaluation Metrics Confusion Matrix, Accuracy
# ------------------------------------------------------------------
# Generalized function with all metrics to run for any algorithm

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return(cm, acc, recall, precision, f1score)


# Model 1 Logistic Regression
lr_result = evaluate_model(y_test, y_pred_lr)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result)

# Model 2 KNN
knn_result = evaluate_model(y_test, y_pred_knn)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result)

# Model 3 Decision Tree
dt_result = evaluate_model(y_test, y_pred_dt)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result)

# Model 4 - Random Forest
rf_result = evaluate_model(y_test, y_pred_rf)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result)

# Model 5 - SVM - Linear
svmlinear_result = evaluate_model(y_test, y_pred_svc)
print("SVM Linear Kernel values for cm, accuracy, recall, precision and f1 score", svmlinear_result)

# Model 6 - SVM - NonLinear
svcrbf_result = evaluate_model(y_test, y_pred_svcrbf)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result)

# Model 7 - Naive Bayes
nb_result = evaluate_model(y_test, y_pred_nb)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result)


# ---------------------------------------------------------------------
# PART B - TOTAL 53 DATA = 42 TRAIN DATA AND 11 TEST DATA
# ---------------------------------------------------------------------
# 10 TEST DATA added from minority class from remaining 47 samples
# which model has not seen
# ---------------------------------------------------------------------
# STEP 1 - Make another test set with more anomaly (class versicolor)
# --------------------------------------------------------------------


# STEP 1A - MAKE Y OR LABELS
#-------------------------------------------------------------------
# Append 10 observations to tail of y_test_xl from new_target_y
y_test_xl = y_test.copy()

# Extract required df from existing df i.e. last 10 rows of class 1
n=90
new_target_y_copy = new_target_y.copy()
new_target_y1_df_copy = pd.DataFrame(new_target_y_copy)
new_target_y1_df_copy.drop(new_target_y1_df_copy.head(n).index,
                      inplace=True)
# Changed back df again to numpy 1D array and flattened
new_target_y1_df_copy = new_target_y1_df_copy.to_numpy().flatten()

# Merge original test dataset and new one with 10 more anomaly added
y_test_merged = np.concatenate((y_test_xl, new_target_y1_df_copy))
print("Merged labels of new test set", y_test_merged)

# STEP 1B - MAKE X OR DATA
#-------------------------------------------------------------------
# Append to X_test_xl the last 10 rows from new_data_X

X_test_xl = X_test.copy()
new_data_X_copy = new_data_X.copy()

# Keep 10 observations from tail of X
n = 10
new_data_X1_copy = new_data_X_copy[-n:, :]
print("Filtered data", new_data_X1_copy)
print("Length of Filtered Data", len(new_data_X1_copy))

# Merge X_test_xl and new_data_X1_copy
X_test_merged = np.concatenate((X_test_xl, new_data_X1_copy))
print("Merged labels of new test set", X_test_merged)


# STEP 2 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset

# Model 1
y_pred_lr1 = logreg.predict(X_test_merged)  # Predict
lr_result1 = evaluate_model(y_test_merged, y_pred_lr1)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result1)

# Model 2
y_pred_knn1 = knn.predict(X_test_merged) # Predict
knn_result1 = evaluate_model(y_test_merged, y_pred_knn1)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result1)

# Model 3
y_pred_dt1 = dt.predict(X_test_merged) # Predict
dt_result1 = evaluate_model(y_test_merged, y_pred_dt1)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result1)

# Model 4
y_pred_rf1 = rf.predict(X_test_merged)  # Predict
rf_result1 = evaluate_model(y_test_merged, y_pred_rf1)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result1)

# Model 5 Linear SVC
y_pred_svc1 = svc.predict(X_test_merged)  # Predict
svmlinear_result1 = evaluate_model(y_test_merged, y_pred_svc1)
print("SVM Linear Kernel values for cm, accuracy, recall, precision and f1 score", svmlinear_result1)

# Model 6 Kernel SVM
y_pred_svcrbf1 = svc_rbf.predict(X_test_merged)  # Predict
svcrbf_result1 = evaluate_model(y_test_merged, y_pred_svcrbf1)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result1)

# Model 7 Naive Bayes
y_pred_nb1 = nbayes.predict(X_test_merged)  # Predict
nb_result1 = evaluate_model(y_test_merged, y_pred_nb1)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result1)



# ---------------------------------------------------------------------
# PART C - TOTAL 53 DATA = 42 TRAIN DATA AND 11 TEST DATA
# ---------------------------------------------------------------------
# 40 TEST DATA added from minority class from remaining 47 samples
# which model has not seen
# ---------------------------------------------------------------------
# STEP 1 - Make another test set with more anomaly (class versicolor)
# --------------------------------------------------------------------

# STEP 1A - MAKE Y OR LABELS
# -------------------------------------------------------------------
# Append 40 observations to tail of y_test_xl from new_target_y
y_test_xl = y_test.copy()

# Extract required df from existing df i.e. last 40 rows of class 1
n = 60
new_target_y_copy = new_target_y.copy()
new_target_y1_df_copy = pd.DataFrame(new_target_y_copy)
new_target_y1_df_copy.drop(new_target_y1_df_copy.head(n).index,
                           inplace=True)
# Changed back df again to numpy 1D array and flattened
new_target_y1_df_copy = new_target_y1_df_copy.to_numpy().flatten()

# Merge original test dataset and new one with 10 more anomaly added
y_test_merged = np.concatenate((y_test_xl, new_target_y1_df_copy))
print("Merged labels of new test set", y_test_merged)

# STEP 1B - MAKE X OR DATA
# -------------------------------------------------------------------
# Append to X_test_xl the last 40 rows from new_data_X

X_test_xl = X_test.copy()
new_data_X_copy = new_data_X.copy()

# Keep 40 observations from tail of X
n = 40
new_data_X1_copy = new_data_X_copy[-n:, :]
print("Filtered data", new_data_X1_copy)
print("Length of Filtered Data", len(new_data_X1_copy))

# Merge X_test_xl and new_data_X1_copy
X_test_merged = np.concatenate((X_test_xl, new_data_X1_copy))
print("Merged labels of new test set", X_test_merged)

# STEP 2 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset

# Model 1
y_pred_lr1 = logreg.predict(X_test_merged)  # Predict
lr_result1 = evaluate_model(y_test_merged, y_pred_lr1)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result1)

# Model 2
y_pred_knn1 = knn.predict(X_test_merged) # Predict
knn_result1 = evaluate_model(y_test_merged, y_pred_knn1)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result1)

# Model 3
y_pred_dt1 = dt.predict(X_test_merged) # Predict
dt_result1 = evaluate_model(y_test_merged, y_pred_dt1)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result1)

# Model 4
y_pred_rf1 = rf.predict(X_test_merged)  # Predict
rf_result1 = evaluate_model(y_test_merged, y_pred_rf1)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result1)

# Model 5 Linear SVC
y_pred_svc1 = svc.predict(X_test_merged)  # Predict
svmlinear_result1 = evaluate_model(y_test_merged, y_pred_svc1)
print("SVM Linear Kernel values for cm, accuracy, recall, precision and f1 score", svmlinear_result1)

# Model 6 Kernel SVM
y_pred_svcrbf1 = svc_rbf.predict(X_test_merged)  # Predict
svcrbf_result1 = evaluate_model(y_test_merged, y_pred_svcrbf1)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result1)

# Model 7 Naive Bayes
y_pred_nb1 = nbayes.predict(X_test_merged)  # Predict
nb_result1 = evaluate_model(y_test_merged, y_pred_nb1)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result1)


