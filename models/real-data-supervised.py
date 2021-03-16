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
# STEP 5 - Modelling
# -----------------------------------------------------------------#
# Input ready for model as in previous step-
# X_train_short_V5 and y_train_short_num for TRAINING MODEL - 8579
# X_test_short_V5 and y_test_short_num for TESTING MODEL - 961

# Step 6 Modelling # 1. Import 2. Instantiate 3. Fit 4. Predict
# --------------------------------------------------------------

# # Model 1 - Logistic Regression
# #------------------------------------------------------------------
#
from sklearn.linear_model import LogisticRegression # Import
logreg = LogisticRegression(max_iter=200) # Instantiate
logreg.fit(X_train_short_V5, y_train_short_num) # Fit
y_pred_lr = logreg.predict(X_test_short_V5) # Predict

# Model 2 - KNN
# ------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier # Import
knn = KNeighborsClassifier(n_neighbors=5) # Instantiate
knn.fit(X_train_short_V5, y_train_short_num) # Fit
y_pred_knn = knn.predict(X_test_short_V5) # Predict

# Model 3 - Decision Tree
# ------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier # Import
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # Instantiate
dt.fit(X_train_short_V5, y_train_short_num) # Fit
y_pred_dt = dt.predict(X_test_short_V5) # Predict

# Model 4 - Random Forest
# ------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier # Import
rf = RandomForestClassifier(n_estimators=200) # Instantiate
rf.fit(X_train_short_V5, y_train_short_num) # Fit
y_pred_rf = rf.predict(X_test_short_V5) # Predict

# Model 5 - SVM - Linear
# ------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc = SVC()  # Instantiate
svc.fit(X_train_short_V5, y_train_short_num)  # Fit
y_pred_svc = svc.predict(X_test_short_V5)  # Predict

# Model 6 - SVM - Non-Linear
# ------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc_rbf = SVC(kernel='rbf', gamma=1.0)  # Instantiate
svc_rbf.fit(X_train_short_V5, y_train_short_num)  # Fit
y_pred_svcrbf = svc_rbf.predict(X_test_short_V5)  # Predict

# Model 7 - Naive Bayes
# ------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB  # Import
nbayes = GaussianNB()  # Instantiate
nbayes.fit(X_train_short_V5, y_train_short_num)  # Fit (features, labels)
y_pred_nb = nbayes.predict(X_test_short_V5)  # Predict


# STEP 7 - Evaluation Metrics
# ------------------------------------------------------------------

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return(cm, acc, recall, precision, f1score)

# Model 1 Logistic Regression

lr_result = evaluate_model(y_test_short_num, y_pred_lr)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result)

# Model 2 - KNN

knn_result = evaluate_model(y_test_short_num, y_pred_knn)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result)

# Model 3 - Decision Tree

dt_result = evaluate_model(y_test_short_num, y_pred_dt)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result)

# Model 4 - Random Forest

rf_result = evaluate_model(y_test_short_num, y_pred_rf)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result)

# Model 5 - SVM - Linear

svclinear_result = evaluate_model(y_test_short_num, y_pred_svc)
print("SVM Linear values for cm, accuracy, recall, precision and f1 score", svclinear_result)

# Model 6 - SVM - Non-Linear

svcrbf_result = evaluate_model(y_test_short_num, y_pred_svcrbf)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result)

# Model 7 - Naive Bayes

nb_result = evaluate_model(y_test_short_num, y_pred_nb)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result)


