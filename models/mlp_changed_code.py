#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:11:59 2020

@author: sylviachadha
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Problem
# Sequence/Pattern and Point Anomaly Detection using Neural Networks
# and time series forecasting approach -detecting unusual shapes and points

# Problem Use Case/Significance
# 1. multiple specific access patterns in intrusion detection
# 2. unusual condition corresponds to an unusual shape in medical conditions

# Reference
# https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/


## PART A : COLLECTIVE/SEQUENCE ANOMALY MODULE
# -------------------------------------------------

# Step 1 Load Dataset and Import Libraries

# --------------------------------------------------
# The more complex the data, the more the analyst has
# to make prior inferences of what is considered normal
# for modeling purposes -  (For sine known pattern)Å


# Import libraries
import pandas as pd
import os
import seaborn as sns
import numpy as np
import statistics
from scipy.stats import norm

# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

# Not to declare index column when reading file
# df = pd.read_csv('sine_new.csv', index_col = 't')

# df = pd.read_csv('sine_new.csv')
df = pd.read_csv('Sine_wave.csv')
df.describe()
df.head()
len(df)

# Split into train and test dataframe


df_train, df_test = np.split(df, [int(0.6 * len(df))])

# Length of dataframes
df_train_len = len(df_train)
df_test_len = len(df_test)

print(df_train_len)
print(df_test_len)

# -------------------------------------------------

# Step 2 Custom Functions Used

# -------------------------------------------------


from numpy import array
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def plot_sine_wave(y, legend):
    ax = y.plot(figsize=(12, 6),
                title='Sine Wave Amplitude = 5, Freq = 10, Period = 0.1sec, Sampling Rate 0.0025sec', label=legend)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    ax.set(xlabel=xlabel, ylabel=ylabel)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    plt.axhline(y=0, color='k')
    return plt


# ------------------------------------------------------------------------

# Step 3 Split into Samples (In prediction these are overlapping samples)
# Both train and test sequence

# ------------------------------------------------------------------------

# train_seq #-----------------------------
train_seq = df_train['y'].to_list()

# choose a number of time steps - to derive using fft
n_steps = 20  ## Assumption for 1 pattern

# Split into samples
train_X, train_Y = split_sequence(train_seq, n_steps)

print(train_X.shape)

# test_seq #-------------------------------
test_seq = df_test['y'].to_list()

# choose a number of time steps
n_steps = 20  ## Assumption for 1 pattern

# Split into samples
test_X, test_Y = split_sequence(test_seq, n_steps)

print(test_X.shape)

# This shape is already in form expected by model [samples,features] so no need to reshape

# --------------------------------------------------------------------

# Step 4 Neural N/W Learning / Model Training only on only Normal data

# ---------------------------------------------------------------------

#  Visualize data with anomaly
zz = plot_sine_wave(df['y'], "")

# Define model

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=n_steps))
model.add(Dense(1))

# Compile model

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Fit model

model.fit(train_X, train_Y, epochs=20, verbose=1)

# ---------------------------------------------------------------------------

# Step 5 Neural N/W Prediction on only Normal Training data

# ---------------------------------------------------------------------------


print(train_X.shape)

# This shape is already in form expected by model [samples,features] so no need to reshape

train_yhat = model.predict(train_X, verbose=0)

print(train_yhat)

# Change prediccted and actual ndarrays to dataframe to see the plot

actual_train = pd.DataFrame(train_Y)

predicted_train = pd.DataFrame(train_yhat)

# Merge two dataframes based on index

mergedDf_train = predicted_train.merge(actual_train, left_index=True, right_index=True)
len(mergedDf_train)
mergedDf_train
print(mergedDf_train.columns)
print(mergedDf_train.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 6A - Qualitative Check
# Plot to see how well predicted train data fits on actual test data
# ---------------------------------------------------------------------------


# Skip n_steps because first prediction starts after n_steps
df_train_new = df_train.iloc[n_steps:]

mergedDf_train.index = df_train_new['t']

# mergedDf.reset_index(inplace=True)
# mergedDf.index += 20


a = mergedDf_train.plot()
plt.show()

# for i in np.arange(start,mergedDf_train.index[-1],n_steps):
#    a.axvline(x=i,color='red',linewidth = 0.5)

# ---------------------------------------------------------------------------
# Step 6B - Quantitative Check
# Mean prediction error for normal window prediction
# Get estimate mean error for all training windows(individually sum errros of each window)
# to compare later with test window anomaly score with 2 S.D from mean error
# ---------------------------------------------------------------------------


# result_train['errors'] = sqrt(mean_squared_error(result_train['actual'] , result_train['predicted']))
# result_train['errors'] = np.mean(np.abs(result_train['predicted'] - result_train['actual']))
# result_train['errors'] = mean_squared_error(result_train['actual'] , result_train['predicted'])

mergedDf_train['errors'] = abs(mergedDf_train['predicted_train'] - mergedDf_train['predicted_train'])
mergedDf_train['errors_raw'] = (mergedDf_train['predicted_train'] - mergedDf_train['actual_train'])

# Show distribution plot of training loss
# Distribution plot cannot be done on absolute error values
mergedDf_train['errors_raw'] = (mergedDf_train['predicted_train'] - mergedDf_train['predicted_train'])

sns.distplot(mergedDf_train['errors_raw'], bins=50, kde=True);
plt.show()

# Sum of prediction errors on sliding windows

total_train_points = len(mergedDf_train['errors'])
total_train_points

total_train_cycles = round(total_train_points / n_steps)
total_train_cycles

n = 0
Cycle_errors = list()
for i in range(1, total_train_cycles):
    i_train_cycle = mergedDf_train['errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_train_cycle)
    Cycle_errors.append(Sum_error_i_Cycle)

print(Cycle_errors)
Mean_window_error = statistics.mean(Cycle_errors)
print("Mean Training window error is", Mean_window_error)

# Window_std = statistics.stdev(Cycle_errors, xbar = Mean_window_error)
# Window_threshold = Mean_window_error + (Window_std*3)


# ---------------------------------------------------------------------------
# Step 7 - Neural N/W Prediction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation - Abnormal pattern will give higher prediction error
# ---------------------------------------------------------------------------


test_yhat = model.predict(test_X, verbose=0)

print(test_yhat)

# Change prediccted and actual ndarrays to dataframe to see the plot

actual_test = pd.DataFrame(test_Y)

predicted_test = pd.DataFrame(test_yhat)

# Merge two dataframes based on index

mergedDf = predicted_test.merge(actual_test, left_index=True, right_index=True)
len(mergedDf)
mergedDf
print(mergedDf.columns)
print(mergedDf.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 8A - Qualitative Check
# Plot to see how well predicted test data fits on actual test data
# ---------------------------------------------------------------------------


# Skip n_steps because first prediction starts after n_steps
df_test_new = df_test.iloc[n_steps:]

mergedDf.index = df_test_new['t']

b = mergedDf.plot()
plt.show()
# for i in np.arange(start,mergedDf.index[-1],n_steps):
#    a.axvline(x=i,color='red',linewidth = 0.5)


# ---------------------------------------------------------------------------
# Step 8B - Quantitative Check
# ---------------------------------------------------------------------------

mergedDf['errors'] = abs(mergedDf['predicted_test'] - mergedDf['actual_test'])
# result_test['errors'] = mean_squared_error(result_test['actual'] , result_test['predicted'])


# Sum of prediction errors on sliding windows


total_test_points = len(mergedDf['errors'])
total_test_points

total_test_cycles = round(total_test_points / n_steps)
total_test_cycles

n = 0
Cycle_count = 0
for i in range(1, total_test_cycles):
    i_test_cycle = mergedDf['errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_test_cycle)
    if Sum_error_i_Cycle > Mean_window_error:
        Cycle_count = Cycle_count + 1
        print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as", Sum_error_i_Cycle)
        print("Anomalous window no ", i)

print('Total Patterns detected ', Cycle_count)

# ---------------------------------------------------------------------------
##PART B - POINT ANOMALY DETECTION MODULE
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step1 - Guassian distribution is assumed for training prediction errors
# Create distribution parameters mean and sd using mle method
# ---------------------------------------------------------------------------


# Show distribution plot of test loss
# Distribution plot cannot be done on absolute error values
mergedDf['errors_raw'] = (mergedDf['predicted_test'] - mergedDf['actual_test'])

sns.distplot(mergedDf['errors_raw'], bins=50, kde=True);
plt.show()

# Use MLE to estimate best parameters for above prediction training errors
parameters = norm.fit(mergedDf['errors_raw'])
print("Mean and S.D obtained using MLE on training prediction errors is", parameters)

mean = parameters[0]
print("Mean", parameters[0])
stdev = parameters[1]
print("S.D", parameters[1])

# ---------------------------------------------------------------------------
# Step 2 - Three Standard Deviation to detect point anomaly on test data
# ---------------------------------------------------------------------------

Threshold_upper = mean + 3 * stdev
print(Threshold_upper)

Threshold_lower = mean - 3 * stdev
print(Threshold_lower)

Point_count = list()
for i in range(len(mergedDf['errors_raw'])):
    if mergedDf['errors_raw'].iloc[i] > Threshold_upper or mergedDf['errors_raw'].iloc[i] < Threshold_lower:
        Point_count.append(mergedDf['errors_raw'].iloc[i])
        print('Anomalous Point Detected at Index', "{:.4f}".format(mergedDf.index[i]), ":",
              "{:.2f}".format(mergedDf['errors_raw'].iloc[i]))
        # print ('Point is at index', result_test.index[i])
print('Total points detected ', len(Point_count))

###-------------------------------------
# Conclusions for MLP Prediction Model
####------------------------------------


# 1. Loss of data of 1 cycle
# 2. Error propagates to the next window (recent data has a lot of effect during prediction time)






