# Aim - ptb-xl load data and then filter for only
# Normal and LVH
# --------------------------------------------------#

import pandas as pd
import numpy as np
import wfdb
import ast
import os


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

# Your code start for preprocess
#----------------------------------------------------#
# PREPROCESS 1 - FILTER DATA INTO NORMAL AND LVH
#----------------------------------------------------#
# Pre-Process data into Normal and LVH only
# Since we get the labels from Y df when we do train-test split
# so we shorten this Y df with only Normal and LVH before splitting
# For shortened dataset we just copy Y to Y_short and then do
# manipulations in Y_short instead of Y

Y_short = Y.copy()

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

# Load filtered signal data (Normal and LVH)
X_short = load_raw_data(Y_short, sampling_rate, path)

#------------------------------------------------------------------
# VERIFICATION FOR PREPROCESS 1
#------------------------------------------------------------------
# In Y ecg id = 30 is lvh and its index is 29
# Check corresponding input signal in X, starts 0.066, 0.269, 0.335
# Now in y_short ecg id = 30 and its index is 9083
# Check corresponding input signal with index 9083 in x_short, starts
# 0.066, 0.269, 0.335

# Verification complete
#------------------------------------------------------------------

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


#----------------------------------------------------#
# PREPROCESS 2 - CONVERT 3D TO 2D (ROWS AND COLUMNS)
#----------------------------------------------------#

df_train = pd.DataFrame()

for i in range(len(X_train_short)):
    # 7th column is V2 lead
    v2_lead = X_train_short[i][:, 7]
    df_temp = pd.DataFrame(v2_lead.reshape(-1, len(v2_lead)))
    df_train = df_train.append(df_temp)

# Check in console - X_train_short[0][:, 7].reshape(-1, 1000)

df_test = pd.DataFrame()

for i in range(len(X_test_short)):
    v2_leadt = X_test_short[i][:, 7]
    df_tempt = pd.DataFrame(v2_leadt.reshape(-1, len(v2_leadt)))
    df_test = df_test.append(df_tempt)

# Now X is dftrain and dftest because we transpose to convert
# to 2 dimensional but their corresponding Y still remains
# y_train_short and y_test short

#------------------------------------------------------------------
# VERIFICATION FOR PREPROCESS 2
#------------------------------------------------------------------
# Verification of df_test from original dataset -------------------

# 1. Check y_test_short[913] = LVH and this is ecg id 1219
# its corresponding input signal is row 913 of df_test as df_test
# is already transposed. Values 0.085, 0.030,-0.035

# 2. Now verify this ecg id from original Y in Y ecg id-1219, label
# should be LVH and index is 1218
# Check corresponding index 1218 for 7th column data in X u get
# same values 0.085,0.030,-0.035

# Verification complete --------------------------------------------
#-------------------------------------------------------------------

# Now since y_train_short and y_test_short are in form ['NORM'] and
# we need label as NORM  before feeding to ml algo we use below command-

y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

## Now ready to input to ML algorithm
# Does order of input matters when distribution of class is ok?????
# Like we obtain Normal and LVH as per stratified sampling already
# but they stack together.
### X - df_train and df_test --> after transpose in form of rows and columns for a single lead
### Y - y_train_short and y_test_short










