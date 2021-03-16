# -------------------------
# STEP 1 Import libraries
# -------------------------

from sklearn.datasets import load_iris
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# STEP 2 Import dataset / Generate Synthetic Data
# ----------------------------------------------------
# Step 2A Generate Synthetic sine wave
# ----------------------------------------------------

#Fs =  # Sampling freq
freq = 0.5 # Inherent freq
amp = 1 # Amplitude
sample = 25000  # For 50 cycles with 500 data points each
time = np.arange(0,100,0.004)

# For loop not required as time already a range function
signal_sine = amp * np.sin(2 * np.pi * freq * time)

# Introduce randomness to data
noise = np.random.normal(0, .1, signal_sine.shape)
new_sine_signal = signal_sine + noise
print(new_sine_signal)

plt.plot(time, new_sine_signal)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()

# ----------------------------------------------------
# Step 2B Generate Synthetic triangle wave
# ----------------------------------------------------

amp1 = 2
freq1 = 0.5

# When u give width=0.5 sawtooth becomes triangle wave
sig_triangle = amp1 * sg.sawtooth(2*np.pi*freq1*time, width=0.5)

# Introduce randomness to data
noise = np.random.normal(0, .1, sig_triangle.shape)
new_triangle_signal = sig_triangle + noise
print(new_triangle_signal)

plt.plot(time, new_triangle_signal)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()


# -------------------------------------------------------------------
# Step 3  Make dataset X_2D(100,500) and y_1D(100) # Features and labels
# -------------------------------------------------------------------
# New merged signal
tot_new_signal = np.concatenate((new_sine_signal, new_triangle_signal))

X_1D = tot_new_signal  # (Since values are going to form the features/X for the model)
X_2D = X_1D.reshape(100, 500)  # Change 1000 data points to 100 rows and 10 columns

# Change to df to see plot
df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and triangle from
# index 50 to 99
df1 = df.iloc[31]
plt.plot(df1)
plt.show()

# y is label
y_class0 = np.zeros(50)
y_class1 = np.ones(50)
y = np.concatenate((y_class0, y_class1))
y_1D = y

# -------------------------------------------------------------------
# Step 4  Reduce dataset X_2D(53,500) and y_1D(53) # Features and labels
# All Class 0 and only 3 samples of Class 1
# Use this 94.3% Class 0 and 5.7% Class 1 for Training
# -------------------------------------------------------------------

# Reduce Triangle class to just 3 samples by removing 47 observations
# from tail of X and Y which are currently 100

n = 47
X_short = X_2D[:-n, :]  # :-n will include all rows starting from 0 upto excluding the last 47 rows
print("Filtered data", X_short)
print("Length of Filtered Data", len(X_short))

# Remove 47 observations from tail of Y
y_short = np.delete(y_1D, range(53, 100, 1))


# -------------------------------------------------------------------
# STEP 5 - Split into Train & Test to feed to ML Algorithm
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
# STEP 1 - Make another test set with more anomaly (class triangle)
# --------------------------------------------------------------------

# Step 1 Make y
y_1D_new = y_1D.copy() # original y which has 100 labels
y_test_xl = y_test.copy()

y_test_xl1 = np.delete(y_1D_new, range(0, 90, 1))

y_test_merged = np.concatenate((y_test_xl, y_test_xl1))
print("Merged labels of new test set", y_test_merged)

# Step 2 Make X
X_2D_new = X_2D.copy()
X_test_xl = X_test.copy()

# Keep 10 observations from tail of X
n = 10
X_2D_new = X_2D_new[-n:, :]
print("Filtered data", X_2D_new)
print("Length of Filtered Data", len(X_2D_new))

# Merge X_test_xl and new_data_X1_copy
X_test_merged = np.concatenate((X_test_xl, X_2D_new))
print("Length of new test set", len(X_test_merged))

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
# STEP 1 - Make another test set with more anomaly (class triangle)
# --------------------------------------------------------------------

# Step 1 Make y
y_1D_new1 = y_1D.copy() # original y which has 100 labels
y_test_xl1 = y_test.copy()

y_test_xl2 = np.delete(y_1D_new1, range(0, 60, 1))

y_test_merged1 = np.concatenate((y_test_xl1, y_test_xl2))
print("Merged labels of new test set", y_test_merged1)

# Step 2 Make X
X_2D_new1 = X_2D.copy()
X_test_xl1 = X_test.copy()

# Keep 40 observations from tail of X
n = 40
X_2D_new1 = X_2D_new1[-n:, :]
print("Filtered data", X_2D_new1)
print("Length of Filtered Data", len(X_2D_new1))

# Merge X_test_xl and new_data_X1_copy
X_test_merged1 = np.concatenate((X_test_xl1, X_2D_new1))
print("Length of new test set", len(X_test_merged1))

# STEP 2 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset

# Model 1
y_pred_lr2 = logreg.predict(X_test_merged1)  # Predict
lr_result2 = evaluate_model(y_test_merged1, y_pred_lr2)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result1)

# Model 2
y_pred_knn2 = knn.predict(X_test_merged1) # Predict
knn_result2 = evaluate_model(y_test_merged1, y_pred_knn2)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result1)

# Model 3
y_pred_dt2 = dt.predict(X_test_merged1) # Predict
dt_result2 = evaluate_model(y_test_merged1, y_pred_dt2)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result1)

# Model 4
y_pred_rf2 = rf.predict(X_test_merged1)  # Predict
rf_result2 = evaluate_model(y_test_merged1, y_pred_rf2)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result1)

# Model 5 Linear SVC
y_pred_svc2 = svc.predict(X_test_merged1)  # Predict
svmlinear_result2 = evaluate_model(y_test_merged1, y_pred_svc2)
print("SVM Linear Kernel values for cm, accuracy, recall, precision and f1 score", svmlinear_result1)

# Model 6 Kernel SVM
y_pred_svcrbf2 = svc_rbf.predict(X_test_merged1)  # Predict
svcrbf_result2 = evaluate_model(y_test_merged1, y_pred_svcrbf2)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result1)

# Model 7 Naive Bayes
y_pred_nb2 = nbayes.predict(X_test_merged1)  # Predict
nb_result2 = evaluate_model(y_test_merged1, y_pred_nb2)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result1)
