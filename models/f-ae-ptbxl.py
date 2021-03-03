# Problem
# Sequence/Pattern Anomaly Detection using Neural Networks
# Reconstruction concept, detecting unusual shapes using Autoencoder

# Problem Use Case/Significance
# Unusual condition corresponds to an unusual shape in medical conditions

# -----------------------------------------------------------------------#
# PART A - Step 1 to 6 - PTB-XL sample code to load data
# -----------------------------------------------------------------------#

# Step 1 - Import libraries
# -------------------------------------------------------------------#
import pandas as pd
import numpy as np
import wfdb
import ast
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


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

# # Split data into train and test ### Original
# # 10 fold stratified sampling
# test_fold = 10
# # Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_subclass
# # Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_subclass

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
    df_train = df_train.append(df_temp,ignore_index=True)

# Check in console - X_train_short[0][:, 7].reshape(-1, 1000)

df_test = pd.DataFrame()

for i in range(len(X_test_short)):
    #   v2_leadt = X_test_short[i][:, 7]
    v2_leadt = X_test_short[i][:, 10]
    df_tempt = pd.DataFrame(v2_leadt.reshape(-1, len(v2_leadt)))
    df_test = df_test.append(df_tempt,ignore_index=True)

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
#df_train_copy.index = np.arange(1, len(df_train_copy) + 1)
df_train_copy.drop(df_train_copy.tail(409).index,inplace=True) # drop last n rows

# TRAIN
# In Sine wave simulation train seq was of length 10 and consist only of
# normal data. In ptb-xl train seq is of length 1000 and consist only of
# normal data.
# When reconstructed fits well to original data

# TEST
# In Sine wave simulation test seq was of length 10 and consist of both
# normal and abnormal data. In ptb-xl train seq is of length 1000 and consist
# of both normal and abnormal data.


# # ---------------------------------------------------------------------------#
# # PART F - Step 14  Custom function for visualization
# # Visualize a. Raw data b. Train data c. Test data ??
# # ---------------------------------------------------------------------------#
#
# # Raw data
# def plot_sine_wave(y, legend):
#     ax = y.plot(figsize=(12, 6),
#                 title='Sine Wave Amplitude = 5, Freq = 10Hz, Period = 0.1sec, Sampling freq-100Hz', label=legend)
#     xlabel = 'time-sec'
#     ylabel = 'amplitude'
#     ax.set(xlabel=xlabel, ylabel=ylabel)
#     plt.axhline(y=0, color='k')
#     return plt
#
# # Visualize data
#
# # A column of a dataframe can be visualized not a row, so you transpose df to
# # to see raw plots
# # df_train_copy(8170,1000) rows*columns
#
# transpose_df = df_train_copy.transpose()
#
# zz = plot_sine_wave(df_train.copy[0], "")
# plt.show()


# ---------------------------------------------------------------------------#
# PART G - Step 15 Autoencoder Neural N/W Learning / Model Training only on only Normal data
# # Steps - a. Define architecture-> b. compile-> c. fit model on training data
# ---------------------------------------------------------------------------#

# 1 shape has 1000 data points so when we feed the sequence of 1000 data points
# Every data point goes as a feature now

# a. Model Architecture
# input layer
input_layer = Input(shape=(1000,))

# encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

# latent view
latent_view = Dense(5, activation='sigmoid')(encode_layer3)

# decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

# output layer
output_layer = Dense(1000)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# b. Model Compilation

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

nb_epoch = 10
learning_rate = 0.1

# c. Model fit on training data (X,X as reconstruction concept)
ae_nn = model.fit(df_train_copy, df_train_copy,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)

# ---------------------------------------------------------#
# Step 16- Predictions on train
# ---------------------------------------------------------#

# Predictions
pred = model.predict(df_train_copy)
pred_transpose = pred.transpose()
df_train_transpose = df_train_copy.transpose()

# ---------------------------------------------------------#
# Step 17 - Qualitative view
# First change ndarray to df inorder to plot
# ---------------------------------------------------------#

pred_transpose_df = pd.DataFrame(pred_transpose)
train_X_transpose_df = pd.DataFrame(df_train_transpose)

mergedDf = pd.DataFrame()

mergedDf["predicted_train"] = pred_transpose_df[0]
mergedDf["actual_train"] = train_X_transpose_df[0]
mergedDf.plot()
plt.show()

mergedDf = pd.DataFrame()
mergedDf["predicted_train"] = pred_transpose_df[1]
mergedDf["actual_train"] = train_X_transpose_df[1]
mergedDf.plot()
plt.show()

mergedDf = pd.DataFrame()
mergedDf["predicted_train"] = pred_transpose_df[2]
mergedDf["actual_train"] = train_X_transpose_df[2]
mergedDf.plot()
plt.show()

## PLOTS
# P1 predited and actual
pred_transpose_df[0].plot()
plt.show()
train_X_transpose_df[1].plot()
plt.show()

# P8170 predited and actual
pred_transpose_df[8169].plot()
plt.show()
train_X_transpose_df[8170].plot()
plt.show()

# mergedDf = pred_transpose_df.merge(train_X_transpose_df, left_index=True, right_index=True)
# print(len(mergedDf))
# mergedDf

# ---------------------------------------------------------#
# Step 18 - Quantitative Check on Training Data
# ---------------------------------------------------------#
# Average mae or mse for the whole training data
# np.average to get 1 value for whole training set
mae_train = np.average(np.abs(mergedDf['actual_train'] - mergedDf['predicted_train']))
mse_train = np.average((mergedDf['actual_train'] - mergedDf['predicted_train']) ** 2)

print("mae for training data is", mae_train)
print("mse for training data is", mse_train)

# Individual absolute error calculation for all points in merged df
mergedDf['window_errors'] = abs(mergedDf['predicted_train'] - mergedDf['actual_train'])

# Sum of prediction errors on sliding/ fixed windows in merged df

n_steps = 1000

total_train_points = len(mergedDf)
print(total_train_points)

total_train_cycles = round(total_train_points / n_steps)
print(total_train_cycles)

# n = 0
# Cycle_errors = list()
# for i in range(1, total_train_cycles):
#     i_train_cycle = mergedDf['window_errors'].iloc[n:n + n_steps]
#     n = n + n_steps
#     Sum_error_i_Cycle = sum(i_train_cycle)
#     Cycle_errors.append(Sum_error_i_Cycle)

Cycle_errors = sum(mergedDf['window_errors'])
print("Cycle_errors", Cycle_errors)

#Mean_window_error = statistics.mean(Cycle_errors)
#print("Mean Training window error is", Mean_window_error)

# Step 19 - Neural N/W reconstruction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation/Hypothesis - Abnormal pattern will give higher reconstruction error
# -------------------------------------------------------------------------------

# Predictions on test set
pred_test = model.predict(df_test)

pred_test_transpose = pred_test.transpose()

df_test_transpose = df_test.transpose()

# Change to df
pred_test_transpose_df = pd.DataFrame(pred_test_transpose)
test_X_transpose_df = pd.DataFrame(df_test_transpose)

# Make a new df merged with above 2 col pred and actual
mergedDf_test = pd.DataFrame()

mergedDf_test["predicted_test"] = pred_test_transpose_df[0]
mergedDf_test["actual_test"] = test_X_transpose_df[0]
mergedDf_test.plot()
plt.show()

mergedDf_test = pd.DataFrame()
mergedDf_test["predicted_test"] = pred_test_transpose_df[960]
mergedDf_test["actual_test"] = test_X_transpose_df[960]
mergedDf_test.plot()
plt.show()

mergedDf_test = pd.DataFrame()
mergedDf_test["predicted_test"] = pred_test_transpose_df[11]
mergedDf_test["actual_test"] = test_X_transpose_df[11]
mergedDf_test.plot()
plt.show()

# ## PLOTS
# # P1 predited and actual
# pred_transpose_df[0].plot()
# plt.show()
# train_X_transpose_df[1].plot()
# plt.show()
#
# # P8170 predited and actual
# pred_transpose_df[8169].plot()
# plt.show()
# train_X_transpose_df[8170].plot()
# plt.show()

# Step 20 - Quantitative Check on Test data
# ---------------------------------------------------------------------------

# Individual absolute error calculation for each test point
mergedDf_test['window_errors'] = abs(mergedDf_test['predicted_test'] - mergedDf_test['actual_test'])

# Sum of prediction errors on fixed windows of step size 10

# From test data preparation we know total subsequences and total data points
# and n_steps
# total_subsequences_test = math.floor(len(test_seq_array) / step_size)
# print("Total_test_subsequences", total_subsequences_test)

# n = 0
# Cycle_count = 0
# for i in range(1, total_subsequences_test):
#     i_test_cycle = mergedDf_test['window_errors'].iloc[n:n + n_steps]
#     n = n + n_steps
#     Sum_error_i_Cycle = sum(i_test_cycle)
#     if Sum_error_i_Cycle > Mean_window_error:  # Condition to declare anomaly
#         Cycle_count = Cycle_count + 1
#         print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as", Sum_error_i_Cycle)
#         print("Anomalous window no ", i)

Cycle_errors = sum(mergedDf_test['window_errors'])
print("Cycle_errors", Cycle_errors)

# print('Total Patterns detected ', Cycle_count)

#--------End --------------------------------------------------------------------------




#-------end ------








# Step 8 Autoencoder Neural N/W Learning / Model Training only on only Normal data
# Steps - a. Define architecture-> b. compile-> c. fit model on training data
# ----------------------------------------------------------------------------------

# 1 shape has 10 data points so when we feed the sequence of 10 data points
# Every data point goes as a feature now

# a. Model Architecture
# input layer
input_layer = Input(shape=(10,))

# encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

# latent view
latent_view = Dense(5, activation='sigmoid')(encode_layer3)

# decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

# output layer
output_layer = Dense(10)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# b. Model Compilation

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

nb_epoch = 30
learning_rate = 0.1

# c. Model fit on training data (X,X as reconstruction concept)
ae_nn = model.fit(train_X, train_X,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)


# Step 9 AE prediction on Training data (which is only normal data)
# ---------------------------------------------------------------------------

# Predicted and actual arrays, need to flatten both because both are sequences of length 10
pred = model.predict(train_X)
pred_train = pred.flatten()

actual_train = train_X.flatten()

print(len(pred_train))
print(len(actual_train))

# Step 10 Change predicted and actual arrays to dataframe to see the plot
# and view actual & predicted values side by side by merging into 1 df
# ---------------------------------------------------------------------------

predicted_df = pd.DataFrame(pred_train)

actual_df = pd.DataFrame(actual_train)

# Merge two dataframes based on index

mergedDf = predicted_df.merge(actual_df, left_index=True, right_index=True)
print(len(mergedDf))
mergedDf
print(mergedDf.columns)
print(mergedDf.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))


# Step 11 - Qualitative Check on Training Data
# Plot to see how well reconstructed data fits on actual data
# ---------------------------------------------------------------------------

# Add index which is df_train['time'] to merged df inorder to plot
mergedDf.index = df_train['time']
mergedDf.plot()
plt.show()

# Step 12 - Quantitative Check on Training Data
# ---------------------------------------------------------------------------
# Average mae or mse for the whole training data
# np.average to get 1 value for whole training set
mae_train = np.average(np.abs(mergedDf['actual_train'] - mergedDf['predicted_train']))
mse_train = np.average((mergedDf['actual_train'] - mergedDf['predicted_train']) ** 2)

print("mae for training data is", mae_train)
print("mse for training data is", mse_train)

# Individual absolute error calculation for all points in merged df
mergedDf['window_errors'] = abs(mergedDf['predicted_train'] - mergedDf['actual_train'])

# Sum of prediction errors on sliding/ fixed windows in merged df

n_steps = 10

total_train_points = len(mergedDf)
print(total_train_points)

total_train_cycles = round(total_train_points / n_steps)
print(total_train_cycles)

n = 0
Cycle_errors = list()
for i in range(1, total_train_cycles):
    i_train_cycle = mergedDf['window_errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_train_cycle)
    Cycle_errors.append(Sum_error_i_Cycle)

print(Cycle_errors)
Mean_window_error = statistics.mean(Cycle_errors)
print("Mean Training window error is", Mean_window_error)

# 3 Standard deviation concept
window_std = statistics.stdev(Cycle_errors, xbar = Mean_window_error)
print(window_std)
window_threshold_u = Mean_window_error + (window_std*3)
window_threshold_l = Mean_window_error - (window_std*3)
print(window_threshold_u)
print(window_threshold_l)


# Step 13 - Neural N/W reconstruction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation/Hypothesis - Abnormal pattern will give higher reconstruction error
# -------------------------------------------------------------------------------

# Predicted and actual arrays, need to flatten both because both are sequences of length 10
pred1 = model.predict(test_X)
pred_test = pred1.flatten()

actual_test = test_X.flatten()

print(len(pred_test))
print(len(actual_test))

# Step 14 - Change predicted and actual arrays to dataframe to see the plot
# ---------------------------------------------------------------------------
predicted_df_test = pd.DataFrame(pred_test)
actual_df_test = pd.DataFrame(actual_test)

# Merge two dataframes based on index

mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
print(len(mergedDf_test))
mergedDf_test
print(mergedDf_test.columns)
print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))


# Step 15 - Qualitative Check on Test Data
# ---------------------------------------------------------------------------
# Plot to see how well predicted test data fits on actual test data
#  new change
if len(df_test) != len(mergedDf_test):
    diff = (len(df_test) - len(mergedDf_test))
    new_df_test = df_test[:-diff]


mergedDf_test.index = new_df_test['time']
mergedDf_test.plot()
plt.show()


# Step 16 - Quantitative Check
# ---------------------------------------------------------------------------

# Individual absolute error calculation for each test point
mergedDf_test['window_errors'] = abs(mergedDf_test['predicted_test'] - mergedDf_test['actual_test'])

# Sum of prediction errors on fixed windows of step size 10

# From test data preparation we know total subsequences and total data points
# and n_steps
# total_subsequences_test = math.floor(len(test_seq_array) / step_size)
# print("Total_test_subsequences", total_subsequences_test)

n = 0
Cycle_count = 0
for i in range(1, total_subsequences_test):
    i_test_cycle = mergedDf_test['window_errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_test_cycle)
    if Sum_error_i_Cycle > Mean_window_error:  # Condition to declare anomaly
        Cycle_count = Cycle_count + 1
        print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as", Sum_error_i_Cycle)
        print("Anomalous window no ", i)

print('Total Patterns detected ', Cycle_count)

#--------End --------------------------------------------------------------------------

