# Aim - ptb-xl load data
# OCSVM - Only normal data in x_train, y_train but both normal & LVH in
# X_test for prediction. # Result comparison
# Test with lead V5 (column 10)
# ------------------------------------------------------------------#

# -----------------------------------------------------------------------#
# PART A - Step 1 to 6 - PTB-XL sample code to load data
# -----------------------------------------------------------------------#

# Step 1 - Import libraries
# -------------------------------------------------------------------#
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# Step 2 - Define function to load raw data
# -------------------------------------------------------------------
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# Define path and sampling rate at which to load data
# -------------------------------------------------------------------
path = '/Users/sylviachadha/Desktop/Thesis/Datasets/ptb-xl/'
sampling_rate = 100

# Step 3 - Read whole database file which has 27 col's including scp_codes
# Load and convert annotation data
# ------------------------------------------------------------------------
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
# ast.literal_eval raises an exception if the input isn't a valid Python
# datatype, so the code won't be executed if it's not
print(Y.columns)

# Step 4 - Load raw signal data
# Reads all files, Y is whole db with 27 columns
# -------------------------------------------------------------------

X = load_raw_data(Y, sampling_rate, path)

# Step 5 - Load scp_statements.csv for diagnostic aggregation
# -------------------------------------------------------------------

agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
# Out of form, rhythm and diagnostic they are separating diagnostic only
agg_df = agg_df[agg_df.diagnostic == 1]


# Step 6 - Adding superclass and subclass to original database
# -------------------------------------------------------------------

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

# -----------------------------------------------------------------------#
# PART B - Step 7 to 9 - Prepare Y_short and X_short with Filter data
# (Normal, LVH)
# -----------------------------------------------------------------------#

# Step 7 - Copy original Y which has all diagnostic class
# -----------------------------------------------------------------------#
Y_short = Y.copy()

# Step 8 - Filter Y_short to only NORM and LVH
# -----------------------------------------------------------------------#

# Below command to change ['NORM'] to NORM
Y_short['clean_class'] = Y_short['diagnostic_subclass'].apply(' '.join)

# If need to reduce number of normal records
# min_norm_df = norm_df.head(3000)
# Y = min_norm_df.append(mi_df, ignore_index=True)

# Extract 2 classes and merge
norm_df = Y_short[Y_short['clean_class'] == 'NORM']
lvh_df = Y_short[Y_short['clean_class'] == 'LVH']
Y_short = norm_df.append(lvh_df)

# If ignore index is true index will be reordered
# Y_short = norm_df.append(lvh_df, ignore_index=True)

# Filter done already, delete column Y['clean_class']
del Y_short['clean_class']

# Step 9 - Load x signals using only filtered Y_short labels NORM & LVH
# -----------------------------------------------------------------------#
# Load signal data (Normal and LVH)
X_short = load_raw_data(Y_short, sampling_rate, path)

# -------------------------------------------#
# PART C - Step 10 - Train-Test Split
# -------------------------------------------#

# Now use same code for Train-Test Split as provided in sample
# but using Y_short and X_short instead of Y and X
# Both shown below -

# Split data into train and test ### Original
# 10 fold stratified sampling
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_subclass

# Split data into train and test  #### Shortened
# 10 fold stratified sampling
test_fold = 10
# Train
X_train_short = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train_short = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test_short = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test_short = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass

print(y_train_short.value_counts())
print(y_test_short.value_counts())

# ---------------------------------------------------------------#
# PART D - Step 12 CONVERT X signals 3D TO 2D (ROWS AND COLUMNS)
# ---------------------------------------------------------------#

df_train = pd.DataFrame()

for i in range(len(X_train_short)):
    # 7th column is V2 lead
    # i will go upto 8579 length of Train ds
    # [:, 10], all rows and only 10th column of each record - 1 lead v2
    v2_lead = X_train_short[i][:, 10]
    # change column to row by flattening using reshape(-1) for len(v2_lead)
    # which is 1000
    df_temp = pd.DataFrame(v2_lead.reshape(-1, len(v2_lead)))
    # every record gets appended one by one
    df_train = df_train.append(df_temp)

# Check in console - X_train_short[0][:, 7].reshape(-1, 1000)

df_test = pd.DataFrame()

for i in range(len(X_test_short)):
    #   v2_leadt = X_test_short[i][:, 7]
    v2_leadt = X_test_short[i][:, 10]
    df_tempt = pd.DataFrame(v2_leadt.reshape(-1, len(v2_leadt)))
    df_test = df_test.append(df_tempt)

# Now X is dftrain and dftest because we transpose to convert
# to 2 dimensional but their corresponding Y still remains
# y_train_short and y_test short


# Now since y_train_short and y_test_short are in form ['NORM'] and
# we need label as NORM
# Change NORM to 1 and 0 before feeding to ml algo we use below command-

y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

# Before feeding to model, we change to numbers
# Independent variables already no's so we convert the labels also
# to numbers
# Decision Tree we need to convert the categorical variable to
# dummy variables (one hot encoding)

# y_train_short_num = pd.get_dummies(y_train_short, drop_first=True)
# y_test_short_num = pd.get_dummies(y_test_short, drop_first=True)


# get_dummies cause series to change to df, so we again use .squeeze()
# to change it back to series
# y_train_short_num = y_train_short_num.squeeze()
# y_test_short_num = y_test_short_num.squeeze()

y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "LVH": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "LVH": 1})

# ---------------------------------------------------------------------------#
# PART E - Step 13  Data Preprocessing specifically for ocsvm
# since we need only the normal class for training x and y, no change in test
# ---------------------------------------------------------------------------#

# Step 13 First check how many 1's in y_train_short_num_copy
# We know train data has 409 LVH and 8170 Normal so we delete the last
# 409 rows from df_train and y_train_short_num_copy

y_train_short_num_copy = y_train_short_num.copy()

# Drop last n rows
y_train_short_num_copy.drop(y_train_short_num_copy.tail(409).index,inplace=True) # drop last n rows
print(y_train_short_num_copy.value_counts())
# Similarly drop last 409 corresponding rows from df_train
# df_train index reflected as 0,0,0,0 so we change it to index 1 to 8579 so
# we can execute .tail(409).index

df_train_copy = df_train.copy()
df_train_copy.index = np.arange(1, len(df_train_copy) + 1)
df_train_copy.drop(df_train_copy.tail(409).index,inplace=True) # drop last n rows

# -------------------------------------------------#
# PART F - Step 14-15   Apply One Class SVM
# -------------------------------------------------#
# Step 14 - SVC declare and fit on train data
# -------------------------------------------------#

# Import OneClassSVM
from sklearn.svm import OneClassSVM

# As this is a class we create an instance of it
# Creating the one class svm

# Define variables
train_rec_count = len(df_train)
print(train_rec_count)
normal_rec_count = len(df_train_copy)
print(normal_rec_count)

nu_percentage = (train_rec_count-normal_rec_count)/train_rec_count
print(nu_percentage)
ocsvm = OneClassSVM(nu=nu_percentage)
# nu: by default, the value is 0.5. It represents the fraction of training errors
# and can be between 0 and 1. Here we specify the anomaly percentage in the
# training dataset.

# ocsvm.fit(X,Y) # Fit on training data
# In our case df_train_copy is a X and y_train_short_num_copy is Y
# as these 2 contain only normal data
ocsvm.fit(df_train_copy, y_train_short_num_copy)

# Step 15 Predict on test data using trained classifier
# -----------------------------------------------------------------------#
# y_predict comes out to be in form of array as expected, no need to
# change in form of series.
y_predict = ocsvm.predict(df_test)
print(y_test_short_num.value_counts())

# ------------------------------------------------------#
# PART F - Step 16   Evaluation Metrics
# ------------------------------------------------------#

# Step 15 Make the confusion matrix for evaluation
# ------------------------------------------------------#

# To evaluate the classifier we must first change the labels in the
# test dataset from 0 and 1 for the majority and minority classes
# respectively, to +1 and -1 or change y_pred 1 -> 0 and -1 -> 1

y_test_short_num_copy = y_test_short_num.copy()
y_test_short_num_copy = y_test_short_num_copy.replace(to_replace={0: 1, 1: -1})


# Confusion Matrix
cm = confusion_matrix(y_test_short_num_copy, y_predict)
print(cm)
print('True negative  [Normal as Normal] = ', cm[1][1])
print('False positive [Normal misclassified as LVH] = ', cm[1][0])
print('True positive  [LVH as LVH] = ', cm[0][0])
print('False negative [LVH misclassified as Normal] = ', cm[0][1])

# plot_confusion_matrix(ocsvm, df_test, y_test_short_num_copy)
# plt.show()

# Actual counts in df using df.value_counts
class_values = (y_test_short_num_copy.value_counts())
print(class_values)

# Step 16 (Accuracy Score)
# ------------------------------------------------------#

# Can use any command below for the accuracy score
# score = dtc.score(df_test, y_test_short_num)
acc_score = accuracy_score(y_test_short_num_copy, y_predict)
print('accuracy_score', acc_score)


# Step 17 (Recall /Sensitivity / Accuracy of LVH/+ve class)
# ------------------------------------------------------#
recall_value = recall_score(y_test_short_num_copy, y_predict,pos_label=-1)
print('recall_value', recall_value)


# Step 18 (Precision - How many +ve's predicted correctly out of all +ve's
# ------------------------------------------------------------------------#
precision_value = precision_score(y_test_short_num_copy, y_predict,pos_label=-1)
print('precision_value', precision_value)


# Step 19 (F1 Score)
# ------------------------------------------------------------------------#
f1_value = f1_score(y_test_short_num_copy, y_predict,pos_label=-1)
print('f1', f1_value)


