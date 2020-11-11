#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 08:14:37 2020

@author: sylviachadha
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

# Problem
# Sequence/Pattern and Point Anomaly Detection using Neural Networks
# and reconstruction concept -detecting unusual shapes and points

# Problem Use Case/Significance
# 1. multiple specific access patterns in intrusion detection
# 2. unusual condition corresponds to an unusual shape in medical conditions


## PART A : COLLECTIVE/SEQUENCE ANOMALY MODULE
# -------------------------------------------------

# Step 1 Load Dataset and Import Libraries

# --------------------------------------------------
# The more complex the data, the more the analyst has
# to make prior inferences of what is considered normal
# for modeling purposes -  (For sine known pattern)


# Import libraries
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
import statistics
from scipy.stats import norm
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model
from tensorflow.python.keras.metrics import mse

# Check working directory

wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

#df = pd.read_csv('sine_synthetic.csv', index_col='t')
df = pd.read_csv('sine_synthetic.csv')
print(df.describe())
print(df.head())
print(len(df))

# Split into train and test dataframe

df_train, df_test = np.split(df, [int(0.7 * len(df))])

# Length of dataframes
df_train_len = len(df_train)
df_test_len = len(df_test)

print(df_train_len)
print(df_test_len)

# -------------------------------------------------

# Step 2 Custom Functions Used

# -------------------------------------------------


def plot_sine_wave(y, legend):
    ax = y.plot(figsize=(12, 6),
                title='Sine Wave Amplitude = 5, Freq = 10Hz, Period = 0.1sec, Sampling freq-100Hz', label=legend)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    ax.set(xlabel=xlabel, ylabel=ylabel)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    plt.axhline(y=0, color='k')
    return plt


# def float_sq(start, stop, step=1):
#     n = int(round((stop - start) / float(step)))
#     if n > 1:
#         return ([start + step * i for i in range(n + 1)])
#     elif n == 1:
#         return ([start])
#     else:
#         return ([])


# Preparing train sequences

#train_seq = df['y'].iloc[:140]

train_seq = df_train['y']
train_seq_array = (train_seq.to_numpy())

print(len(train_seq))
print(len(train_seq_array))

n = 0
step_size = 10

total_subsequences = math.floor(len(train_seq_array) / step_size)
print("Total_train_subsequences", total_subsequences)
total_data_points = total_subsequences * step_size
print("Total_train_points", total_data_points)

array_tuple = []
while n < total_data_points:
    subsequence = (train_seq_array[n:n + step_size])
    print(subsequence)
    n = n + step_size
    print(n)
    array_tuple.append(subsequence)

# Combining all sequences into 1 single train array
# array_tuple = (array1, array2, array3)
# arrays = np.vstack(array_tuple)

train_X = np.vstack(array_tuple)

# =============================================================================
# Preparing test sequences

#test_seq = df['y'].iloc[140:]
test_seq = df_test['y']
test_seq_array = (test_seq.to_numpy())

print(len(test_seq))
print(len(test_seq_array))
# Need to ensure that it is taking first 400 points only

n = 0
step_size = 10

total_subsequences_test = math.floor(len(test_seq_array) / step_size)
print("Total_test_subsequences", total_subsequences_test)
total_data_points_test = total_subsequences_test * step_size
print("Total_test_points", total_data_points_test)

# So choose to create test sequences only with those many points


array_tuple = []
while n < total_data_points_test:
    subsequence_test = (test_seq_array[n:n + step_size])
    print(subsequence_test)
    n = n + step_size
    print(n)
    array_tuple.append(subsequence_test)

test_X = np.vstack(array_tuple)
# =============================================================================

#  Visualize data with anomaly
zz = plot_sine_wave(df['y'], "")
plt.show()
# ----------------------------------------------------------------------------------

# Step 3 Autoencoder Neural N/W Learning / Model Training only on only Normal data
# Steps - 1. Define architecture-> 2. compile-> 3. fit model on training data
# Autoencoder neural network architecture

# -----------------------------------------------------------------------å-----------

# 1 shape has 10 data points so when we feed the sequence of 10 data points
# each data point will be treated as a column, so 10 columns will be there.


## input layer
input_layer = Input(shape=(10,))

## encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

## latent view
latent_view = Dense(5, activation='sigmoid')(encode_layer3)

## decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

## output layer
output_layer = Dense(10)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# Model Compilation

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

nb_epoch = 30
learning_rate = 0.1

ae_nn = model.fit(train_X, train_X,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)

# ---------------------------------------------------------------------------

# Step 4 Neural N/W reconstruction on only Normal Training data

# ---------------------------------------------------------------------------

# Predicted and actual arrays, need to flatten both because both are sequences of length 40
pred = model.predict(train_X)
pred_train = pred.flatten()

actual_train = train_X.flatten()

print(len(pred_train))
print(len(actual_train))

# Change predicted and actual arrays to dataframe to see the plot

predicted_df = pd.DataFrame(pred_train)

actual_df = pd.DataFrame(actual_train)

# Merge two dataframes based on index

mergedDf = predicted_df.merge(actual_df, left_index=True, right_index=True)
print(len(mergedDf))
mergedDf
print(mergedDf.columns)
print(mergedDf.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# Below df replicated for point anomalies

mergedDf1 = predicted_df.merge(actual_df, left_index=True, right_index=True)
len(mergedDf1)
mergedDf1
print(mergedDf1.columns)
print(mergedDf1.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 5A - Qualitative Check
# Plot to see how well reconstructed data fits on actual data
# ---------------------------------------------------------------------------

#from matplotlib.pyplot import *

mergedDf.index = df_train['t']


mergedDf.plot()
plt.show()


# ---------------------------------------------------------------------------
# Step 5B - Quantitative Check
# Mean prediction error for normal window reconstruction
# Get estimate mean error for all training windows(individually sum errros of each window)
# to compare later with test window anomaly score with 2 S.D from mean error
# ---------------------------------------------------------------------------

# Average mae or mse for the whole training data
mae_train = np.average(np.abs(mergedDf['actual_train'] - mergedDf['predicted_train']))

mse_train = np.average((mergedDf['actual_train'] - mergedDf['predicted_train']) ** 2)

print("mae for training data is", mae_train)
print("mse for training data is", mse_train)

# Individual absolute error calculation for each window

mergedDf['window_errors'] = abs(mergedDf['predicted_train'] - mergedDf['actual_train'])

# Make assumption of normal distribution on point errors
mergedDf1['point_errors'] = (mergedDf1['predicted_train'] - mergedDf1['actual_train'])
# Distribution plot of point errors
sns.distplot(mergedDf1['point_errors'], bins=50, kde=True);
plt.show()

# Sum of prediction errors on sliding windows

n_steps = 10

total_train_points = len(mergedDf['window_errors'])
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

# 3 Standard deviation concept ##3******* to explore
window_std = statistics.stdev(Cycle_errors, xbar = Mean_window_error)
print(window_std)
window_threshold_u = Mean_window_error + (window_std*3)
window_threshold_l = Mean_window_error - (window_std*3)
print(window_threshold_u)
print(window_threshold_l)

# ---------------------------------------------------------------------------
# Step 6 - Neural N/W reconstruction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation - Abnormal pattern will give higher reconstruction error
# ---------------------------------------------------------------------------


# Predicted and actual arrays, need to flatten both because both are sequences of length 10
pred1 = model.predict(test_X)
pred_test = pred1.flatten()

actual_test = test_X.flatten()

print(len(pred_test))
print(len(actual_test))

# Change prediccted and actual arrays to dataframe to see the plot

predicted_df_test = pd.DataFrame(pred_test)

actual_df_test = pd.DataFrame(actual_test)

# Merge two dataframes based on index

mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
print(len(mergedDf_test))
mergedDf_test
print(mergedDf_test.columns)
print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))

# Dataframe to detect point anomalies

mergedDf1_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
len(mergedDf1_test)
mergedDf1_test
print(mergedDf1_test.columns)
print(mergedDf1_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 6A - Qualitative Check
# Plot to see how well predicted test data fits on actual test data
# ---------------------------------------------------------------------------
#  new change
if len(df_test) != len(mergedDf_test):
    diff = (len(df_test) - len(mergedDf_test))
    new_df_test = df_test[:-diff]

mergedDf_test.index = new_df_test['t']

mergedDf_test.plot()
plt.show()

# ---------------------------------------------------------------------------
# Step 6B - Quantitative Check
# ---------------------------------------------------------------------------


# Individual absolute error calculation for each test window

mergedDf_test['window_errors'] = abs(mergedDf_test['predicted_test'] - mergedDf_test['actual_test'])

# Make assumption of normal distribution on point errors
mergedDf1_test['point_errors'] = (mergedDf1_test['predicted_test'] - mergedDf1_test['actual_test'])  # Plotting
#mergedDf1_test['point_errors'] = mse(mergedDf1_test['predicted_test'] - mergedDf1_test['actual_test'])
#mergedDf1_test['point_errors'] = np.square(np.subtract(mergedDf1_test['actual_test'],mergedDf1_test['predicted_test'])).mean()

# Distribution plot of point errors
# sns.distplot(mergedDf1_test['point_errors'], bins=50, kde=True);
# plt.show()

# Sum of prediction errors on sliding windows

# From test data preparation we know total subsequences and total data points
# and n_steps
# total_subsequences =  math.floor(len(test_seq_array)/step_size)
# print("Total_train_subsequences", total_subsequences)
# total_data_points = total_subsequences * step_size
# print("Total_train_points",total_data_points)


# n = 0
# Cycle_errors = list()
# for i in range(1, total_subsequences):
#     per_subsequences_points = mergedDf['errors'].iloc[n:n + n_steps]
#     n = n + n_steps
#     Sum_error_per_subsequences = sum(per_subsequences_points)
#     Cycle_errors.append(Sum_error_per_subsequences)

# Sum of prediction errors on sliding windows

n = 0
Cycle_count = 0
for i in range(1, total_subsequences_test):
    i_test_cycle = mergedDf_test['window_errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_test_cycle)
    if Sum_error_i_Cycle > Mean_window_error:
        Cycle_count = Cycle_count + 1
        print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as", Sum_error_i_Cycle)
        print("Anomalous window no ", i)

print('Total Patterns detected ', Cycle_count)

# #---------------------------------------------------------------------------
# #PART B - SEPERATE POINT ANOMALY DETECTION MODULE BASED ON PREDICTION ERRORS
# #---------------------------------------------------------------------------

# #---------------------------------------------------------------------------
# # Step1 - Guassian distribution is assumed for training prediction errors
# # Create distribution parameters mean and sd using mle method
#  #---------------------------------------------------------------------------

sns.distplot(mergedDf1_test['point_errors'], bins=50, kde=True);
plt.show()

parameters = norm.fit(mergedDf1['point_errors'])
# print ("Mean and S.D obtained using MLE on training prediction errors is",parameters)


mean = parameters[0]
print("Mean", parameters[0])
stdev = parameters[1]
print("S.D", parameters[1])

# #---------------------------------------------------------------------------
# # Step 2 - Three Standard Deviation to detect point anomaly on test data
# #---------------------------------------------------------------------------

Threshold_p = mean + 3 * stdev
Threshold_n = mean - 3 * stdev
print(Threshold_p)
print(Threshold_n)

Point_count = list()
for i in range(len(mergedDf1_test['point_errors'])):
    if mergedDf1_test['point_errors'].iloc[i] > Threshold_p or mergedDf1_test['point_errors'].iloc[i] < Threshold_n:
        Point_count.append(mergedDf1_test['point_errors'].iloc[i])
        print('Anomalous Point Detected at Index', "{:.4f}".format(mergedDf1_test.index[i]), ":",
              "{:.2f}".format(mergedDf1_test['point_errors'].iloc[i]))
        # print ('Point is at index', result_test.index[i])
print('Total points detected ', len(Point_count))

# ----------------------------------------------------------------------------
##PART C - CATEGORIZATION MODULE - SEQUENCE OR WINDOW ANOMALY (OVERLAPPING)
# ----------------------------------------------------------------------------

# Decide if whole window anomalous or not (no of points detected as anomalies
# and location of anomalies - if occur together)

# Decide if individual point is anomalouys or not -

# Step1 - Assume a normal distribution on prediction errors
















