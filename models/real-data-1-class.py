# Aim - ptb-xl load data and then filter for only
# Normal and LVH
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#

import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylviachadha/Desktop/Thesis/Datasets/ptb-xl/'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


def aggregate_diagnostic1(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)


# ---------------------------------------------------------------#
# STEP 2 - Extracting Class of Anomaly out of 71 classes in Y
# ---------------------------------------------------------------#
# Once original X and y are loaded u shorten the y to NORMAL
# and ANOMALY class u want and then load corresponding X again

# Add clean class
Y_short = Y.copy()
Y_short['clean_class'] = Y_short['diagnostic_subclass'].apply(' '.join)

# Make a short df with anomaly and normal class
Y_lvh = Y_short[Y_short["clean_class"] == "LVH"]
Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_lvh]
Y_short = pd.concat(frames)

# Check counts of normal and anomaly class
value_counts = Y_short['clean_class'].value_counts()
print(value_counts)

# Since Filtering and value counts done, remove the clean class
del Y_short['clean_class']

# Load corresponding X as per Y_short
X_short = load_raw_data(Y_short, sampling_rate, path)


# -----------------------------------------------------------------#
# STEP 3 - Train/Test Split   (# AS PAPER)
# -----------------------------------------------------------------#
# 10 fold stratified sampling
test_fold = 10
# Train
X_train_short = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train_short = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test_short = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test_short = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass


# -----------------------------------------------------------------#
# STEP 4 - Extracting Channel / Lead out of 12 leads
# Preparing X for the model
# -----------------------------------------------------------------#
# Change X from 3D to 2D with only Lead V5 Signal for each patient

print(X_train_short.shape)
print(X_test_short.shape)
# Select only Lead V5 from X which is column 10
# Leads (I, II, III, AVL, AVR, AVF, V1, ..., V6)

X_train_short_V5 = X_train_short[:, :, 10]
X_test_short_V5 = X_test_short[:, :, 10]


# Before feed to model need to change y_train_short & y_test_short
# labels from ['NORM'] to NORM

# -----------------------------------------------------------------#
# STEP 5 - Changing ['NORM'] to NORM and further to 0 (encoding)
# Preparing y for the model
# -----------------------------------------------------------------#

# Change ['NORM'] to NORM - since applying to series no need to specify
# column names
y_train_short.index.name
y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

# Replace NORM with 0 and Anomaly Class with 1
y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "LVH": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "LVH": 1})

testlabel_counts = y_test_short_num.value_counts()
print(testlabel_counts)
trainlabel_counts = y_train_short_num.value_counts()
print(trainlabel_counts)


# -----------------------------------------------------------------#
# STEP 6 - One CLass methods / Unsupervised methods preprocessing
# -----------------------------------------------------------------#
# Require only normal data for training but require both normal &
# anomalous data for test

# X_train_short_V5 and y_train_short_num - Remove Class 1
# Can be done using below line - no need to remove, just pick out of all
# X_train = X_train[y_train == 0], u only need X_train since one-class
# algorithm fits only on X_train (there is no second class so do not need
# the labels)

X_train_short_V5_1class = X_train_short_V5[y_train_short_num == 0]

# -----------------------------------------------------------------#
# STEP 7 - Modelling #1. Import 2.Instantiate  3. Fit  4. Predict
# -----------------------------------------------------------------#
# Input ready for model as in previous step-
# X_train_short_V5_1class

# Model 1 # One Class SVM
# --------------------------------------------------------------

# A. Import
from sklearn.svm import OneClassSVM

# B. Instantiate
ocsvm = OneClassSVM(gamma='scale', nu=0.05)

# C. Train
ocsvm.fit(X_train_short_V5_1class)

# D. Predict
ypred1 = ocsvm.predict(X_test_short_V5)
#cprint("y_pred", ypred1)

# Model 2 # Isolation Forest
# --------------------------------------------------------------

# A. Import
from sklearn.ensemble import IsolationForest

# B. Instantiate
iforest = IsolationForest(contamination=0.05)

# C. Train on Majority class only
iforest.fit(X_train_short_V5_1class)

# D. Predict
ypred2 = iforest.predict(X_test_short_V5)
# print("y_pred", ypred2)


# -----------------------------------------------------------------------------
# Step 6 - O/P of 1 class - Preprocessing before evaluation
# -----------------------------------------------------------------------------
# Change actual y_test to 1 & -1 as y_pred is returned in this form
# Mark inliers 1, outliers -1 in test_y


y_test_short_num[y_test_short_num == 1] = -1
y_test_short_num[y_test_short_num == 0] = 1
# print("y_test", y_test_short_num)

# -----------------------------------------
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
ocsvm_result = evaluate_model(y_test_short_num, ypred1)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result)
ocsvm.report = classification_report(y_test_short_num, ypred1)
print(ocsvm.report)

# Model 2 Isolationn Forest
iforest_result = evaluate_model(y_test_short_num, ypred2)
print("Isolation Forest values for cm, accuracy, recall, precision and f1 score", iforest_result)
iforest.report = classification_report(y_test_short_num, ypred2)
print(iforest.report)