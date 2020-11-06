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


# PART A : COLLECTIVE/SEQUENCE ANOMALY MODULE
# -------------------------------------------------

# Step 1 Load Dataset and Import Libraries

# --------------------------------------------------
# The more complex the data, the more the analyst has 
# to make prior inferences of what is considered normal 
# for modeling purposes -  (For sine known pattern)


# Import libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from numpy import array

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
from scipy.stats import norm
from scipy.stats import lognorm
from scipy import stats
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
# from outliers import smirnov_grubbs as grubbs
from keras.layers import Input, Dense, Dropout
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

df = pd.read_csv('Sine_wave.csv', index_col='time')
df.describe()
df.head()
len(df)

# -------------------------------------------------

# Step 2 Custom Functions Used

# -------------------------------------------------


from numpy import array
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os


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


def float_sq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    elif n == 1:
        return ([start])
    else:
        return ([])


# Preparing train sequences

train_seq = df['y'].iloc[:600]
train_seq_array = (train_seq.to_numpy())

len(train_seq)
len(train_seq_array)

import math
import numpy as np

n = 0
step_size = 40

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

test_seq = df['y'].iloc[600:]
test_seq_array = (test_seq.to_numpy())

len(test_seq)
len(test_seq_array)
# Need to ensure that it is taking first 400 points only


import math
import numpy as np

n = 0
step_size = 40

total_subsequences = math.floor(len(test_seq_array) / step_size)
print("Total_train_subsequences", total_subsequences)
total_data_points = total_subsequences * step_size
print("Total_train_points", total_data_points)

# So choose to create test sequences only with those many points


array_tuple = []
while n < total_data_points:
    subsequence = (test_seq_array[n:n + step_size])
    print(subsequence)
    n = n + step_size
    print(n)
    array_tuple.append(subsequence)

test_X = np.vstack(array_tuple)
# =============================================================================

#  Visualize data with anomaly
zz = plot_sine_wave(df['y'], "")

# ----------------------------------------------------------------------------------

# Step 3 Autoencoder Neural N/W Learning / Model Training only on only Normal data
# Steps - 1. Define architecture-> 2. compile-> 3. fit model on training data
# Autoencoder neural network architecture

# -----------------------------------------------------------------------å-----------

# 1 shape has 40 data points so when we feed the sequence of 40 data points 
# each data point will be treated as a column, so 40 columns will be there.


## input layer
input_layer = Input(shape=(40,))

## encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

## latent view
latent_view = Dense(10, activation='sigmoid')(encode_layer3)

## decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

## output layer
output_layer = Dense(40)(decode_layer3)

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
preds = model.predict(train_X)
pred_train = preds.flatten()

actual_train = train_X.flatten()

len(pred_train)
len(actual_train)

# Change prediccted and actual arrays to dataframe to see the plot

predicted_df = pd.DataFrame(pred_train)

actual_df = pd.DataFrame(actual_train)

# Merge two dataframes based on index

mergedDf = predicted_df.merge(actual_df, left_index=True, right_index=True)
len(mergedDf)
mergedDf
print(mergedDf.columns)
print(mergedDf.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 5A - Qualitative Check
# Plot to see how well reconstructed data fits on actual data
# ---------------------------------------------------------------------------

from matplotlib.pyplot import *

mergedDf.plot()

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

mergedDf['errors'] = abs(mergedDf['predicted_train'] - mergedDf['actual_train'])

# Sum of prediction errors on sliding windows

n_steps = 40

total_train_points = len(mergedDf['errors'])
total_train_points

total_train_cycles = round(total_train_points / n_steps)
total_train_cycles

n = 0
Cycle_errors = list()
for i in range(1, total_train_cycles):
    i_train_cycle = mergedDf['errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_train_cycle)
    Cycle_errors.append(Sum_error_i_Cycle)

print(Cycle_errors)
Mean_window_error = statistics.mean(Cycle_errors)
print("Mean Training window error is", Mean_window_error)

# 3 Standard deviation concept ##3******* to explore
# window_std = statistics.stdev(Cycle_errors, xbar = Mean_window_error)
# window_threshold = Mean_window_error + (window_std*3)


# ---------------------------------------------------------------------------
# Step 6 - Neural N/W reconstruction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation - Abnormal pattern will give higher reconstruction error 
# ---------------------------------------------------------------------------


# Predicted and actual arrays, need to flatten both because both are sequences of length 40
pred1 = model.predict(test_X)
pred_test = pred1.flatten()

actual_test = test_X.flatten()

len(pred_test)
len(actual_test)

# Change prediccted and actual arrays to dataframe to see the plot

predicted_df_test = pd.DataFrame(pred_test)

actual_df_test = pd.DataFrame(actual_test)

# Merge two dataframes based on index

mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
len(mergedDf_test)
mergedDf_test
print(mergedDf_test.columns)
print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 6A - Qualitative Check
# Plot to see how well predicted test data fits on actual test data
# ---------------------------------------------------------------------------


from matplotlib.pyplot import *

mergedDf_test.plot()

# ---------------------------------------------------------------------------
# Step 6B - Quantitative Check
# ---------------------------------------------------------------------------


# Individual absolute error calculation for each test window

mergedDf_test['errors'] = abs(mergedDf_test['predicted_test'] - mergedDf_test['actual_test'])

# Sum of prediction errors on sliding windows

# From test data preparation we know total subsequences and total data points
# and n_steps 
# total_subsequences =  math.floor(len(test_seq_array)/step_size)
# print("Total_train_subsequences", total_subsequences)
# total_data_points = total_subsequences * step_size
# print("Total_train_points",total_data_points)

n = 0
Cycle_errors = list()
for i in range(1, total_subsequences):
    per_subsequences_points = mergedDf['errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_per_subsequences = sum(per_subsequences_points)
    Cycle_errors.append(Sum_error_per_subsequences)

# Sum of prediction errors on sliding windows

n = 0
Cycle_count = 0
for i in range(1, total_subsequences):
    i_test_cycle = mergedDf_test['errors'].iloc[n:n + n_steps]
    n = n + n_steps
    Sum_error_i_Cycle = sum(i_test_cycle)
    if Sum_error_i_Cycle > Mean_window_error:
        Cycle_count = Cycle_count + 1
        print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as", Sum_error_i_Cycle)
        print("Anomalous window no ", i)

print('Total Patterns detected ', Cycle_count)

# -------------------------------------------------------------------
##PART B - CATEGORIZATION MODULE - SEQUENCE OR WINDOW ANOMALY
# -------------------------------------------------------------------

# Decide if whole window anomalous or not (no of points detected as anomalies
# and location of anomalies - if occur together)

# Decide if individual point is anomalouys or not -

# Step1 - Assume a normal distribution on prediction errors


# #---------------------------------------------------------------------------
# #PART B - SEPERATE POINT ANOMALY DETECTION MODULE BASED ON PREDICTION ERRORS
# #---------------------------------------------------------------------------      

# #---------------------------------------------------------------------------
# # Step1 - Guassian distribution is assumed for training prediction errors
# # Create distribution parameters mean and sd using mle method
#  #---------------------------------------------------------------------------       

# result_train['errors']


# # Show distribution plot of training loss
# sns.distplot(result_train['errors'], bins=50, kde=True);
# plt.show()


# # Use MLE to estimate best parameters for above prediction training errors
# parameters = norm.fit(result_train['errors'])
# print ("Mean and S.D obtained using MLE on training prediction errors is",parameters)


# mean = parameters[0]      
# print ("Mean",parameters[0])
# stdev = parameters[1]   
# print ("S.D", parameters[1])

# #---------------------------------------------------------------------------
# # Step 2 - Three Standard Deviation to detect point anomaly on test data
# #---------------------------------------------------------------------------

# Threshold = mean + 3*stdev
# print(Threshold)


# Point_count = list()
# for i in range(len(result_test['errors'])):
#     if result_test['errors'].iloc[i] > Threshold:
#         Point_count.append(result_test['errors'].iloc[i])
#         print('Anomalous Point Detected at Index',  "{:.4f}".format(result_test.index[i]), ":" ,"{:.2f}".format(result_test['errors'].iloc[i]))
#         #print ('Point is at index', result_test.index[i])
# print('Total points detected ', len(Point_count))


# # Grubbs Test for outlier detection

# # Grubbs.max_test_indices(data, alpha=.05)(result_test['errors'], alpha=.05)
