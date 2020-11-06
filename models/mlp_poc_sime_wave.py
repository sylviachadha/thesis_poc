#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Problem
# Sequence/Pattern and Point Anomaly Detection using Neural Networks
# and time series forecating approach -detecting unusual shapes and points

# Problem Use Case/Significance
# 1. multiple specific access patterns in intrusion detection 
# 2. unusual condition corresponds to an unusual shape in medical conditions


## PART A : COLLECTIVE/SEQUENCE ANOMALY MODULE
#-------------------------------------------------

# Step 1 Load Dataset and Import Libraries

#--------------------------------------------------
# The more complex the data, the more the analyst has 
# to make prior inferences of what is considered normal 
# for modeling purposes -  (For sine known pattern)Å


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
#from outliers import smirnov_grubbs as grubbs

# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')


df = pd.read_csv('Sine_wave.csv', index_col = 'time')
df.describe()
df.head()
len(df)

df_anomaly = pd.read_csv('Sine_wave_anomaly.csv')
anomaly = df_anomaly[df_anomaly['Label'] == 1]

#-------------------------------------------------

# Step 2 Custom Functions Used

#-------------------------------------------------


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


def plot_sine_wave(y,legend):
    ax = y.plot(figsize=(12, 6),
                title='Sine Wave Amplitude = 5, Freq = 10, Period = 0.1sec, Sampling Rate 0.0025sec',label=legend)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    ax.set(xlabel=xlabel, ylabel=ylabel)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    plt.axhline(y=0, color='k')
    return plt


# Model takes no of steps as its dimensions
# Technically, the model will view each time step as a separate feature instead of separate time steps.
# The model expects the input shape to be two-dimensional with [samples, features]
def get_model(n_steps):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=['mse'])
    return model


def float_sq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    elif n == 1:
        return ([start])
    else:
        return ([])
#---------------------------------------------------------------------

# Step 3 Neural N/W Learning / Model Training only on only Normal data

#---------------------------------------------------------------------

#  Visualize data with anomaly
zz = plot_sine_wave(df['y'],"")

#zz.scatter(anomaly['time'], anomaly['y'],c='r',label='Point anomalies')
zz.show()

train_seq = df['y'].iloc[:601].to_list()

# choose a number of time steps
n_steps = 40 ## Assumption for 1 pattern

# Split into samples
train_X, train_Y = split_sequence(train_seq, n_steps)

# Define model
model = get_model(n_steps)
model.fit(train_X, train_Y, epochs=50, verbose=1)


#---------------------------------------------------------------------------

# Step 4 Neural N/W Prediction on only Normal Training data 

#---------------------------------------------------------------------------
#The model expects the input shape to be two-dimensional with [samples, features]
# We have 561 samples

predicted = list()

for x in train_X:
    x_input = array(x)
    x_input = x_input.reshape((1, n_steps))
    yhat = model.predict(x_input, verbose=0)
    predicted.append(yhat.flatten())

result_train = pd.DataFrame(np.concatenate(predicted), columns=["predicted"])
result_train['actual'] = np.array(train_Y)
result_train.index = float_sq(0.1, 1.5, 0.0025)
#result_test.index = float_sq(1.6025, 2.5, 0.0025)

#---------------------------------------------------------------------------
# Step 5A - Qualitative Check
# Plot to see how well predicted data fits on actual data
#---------------------------------------------------------------------------


xx=plot_sine_wave(result_train['actual'],"actual")

plot_sine_wave(result_train['predicted'],"predicted")
for i in np.arange(0.0, 1.5, 0.1):
    xx.axvline(x=i,color='red',linewidth = 0.5)
    
xx.legend(loc="upper right")

xx.show()



#result_train.plot()

#---------------------------------------------------------------------------
# Step 5B - Quantitative Check
# Mean prediction error for normal window prediction
# Get estimate mean error for all training windows(individually sum errros of each window) 
# to compare later with test window anomaly score with 2 S.D from mean error
#---------------------------------------------------------------------------

#result_train['errors'] = sqrt(mean_squared_error(result_train['actual'] , result_train['predicted']))
result_train['errors'] = abs(result_train['predicted'] - result_train['actual'])
#result_train['errors'] = np.mean(np.abs(result_train['predicted'] - result_train['actual']))
#result_train['errors'] = mean_squared_error(result_train['actual'] , result_train['predicted'])



# Sum of prediction errors on sliding windows

total_train_points = len(result_train['errors'])
total_train_points

total_train_cycles = round(total_train_points/n_steps)
total_train_cycles

n=0
Cycle_errors = list()
for i in range(1,total_train_cycles):
    i_train_cycle = result_train['errors'].iloc[n:n+n_steps]
    n = n+n_steps
    Sum_error_i_Cycle = sum(i_train_cycle)
    Cycle_errors.append(Sum_error_i_Cycle)
    
    
print(Cycle_errors)
Mean_window_error = statistics.mean(Cycle_errors)
print ("Mean Training window error is" , Mean_window_error)

#Window_std = statistics.stdev(Cycle_errors, xbar = Mean_window_error)
#Window_threshold = Mean_window_error + (Window_std*3)

#---------------------------------------------------------------------------
# Step 6 - Neural N/W Prediction on Test Data which contains Normal as well
# as abnormal pattern
# Expectation - Abnormal pattern will give higher prediction error 
#---------------------------------------------------------------------------


test_seq = df['y'].iloc[601:].to_list()
test_X, test_Y = split_sequence(test_seq, n_steps)
predicted = list()

for x in test_X:
    x_input = array(x)
    x_input = x_input.reshape((1, n_steps))
    yhat = model.predict(x_input, verbose=0)
    predicted.append(yhat.flatten())

result_test = pd.DataFrame(np.concatenate(predicted), columns=["predicted"])
result_test['actual'] = np.array(test_Y)
result_test.index = float_sq(1.6025, 2.5, 0.0025)

#---------------------------------------------------------------------------
# Step 6A - Qualitative Check
# Plot to see how well predicted test data fits on actual test data
#---------------------------------------------------------------------------

xx=plot_sine_wave(result_test['actual'],'actual')

plot_sine_wave(result_test['predicted'],'predicted')

for i in np.arange(1.5, 2.6, 0.1):
    xx.axvline(x=i,color='red',linewidth = 0.5)

xx.legend(loc="upper right")

xx.show()


#test_plot.axhline(y=0, color='k',linewidth = 1)

#result_test.plot()
#for i in range(0, len(result_test), n_steps): 
#    plt.axvline(x=i, color='red',linewidth = 0.5)
    

#xcoords=list()
#result_test.plot()

#xcoords = [30, 60, 90,120,150,180,210,240,270,300,330,360]
#for xc in xcoords:
#    plt.axvline(x=xc)

#---------------------------------------------------------------------------
# Step 6B - Quantitative Check
#---------------------------------------------------------------------------

result_test['errors'] = abs(result_test['predicted'] - result_test['actual'])
#result_test['errors'] = mean_squared_error(result_test['actual'] , result_test['predicted'])


# Sum of prediction errors on sliding windows


total_test_points = len(result_test['errors'])
total_test_points

total_test_cycles = round(total_test_points/n_steps)
total_test_cycles

n=0
Cycle_count = 0
for i in range(1,total_test_cycles):
    i_test_cycle = result_test['errors'].iloc[n:n+n_steps]
    n = n+n_steps
    Sum_error_i_Cycle = sum(i_test_cycle)
    if Sum_error_i_Cycle > Mean_window_error:
        Cycle_count = Cycle_count + 1
        print("Anomalous Pattern Detected with sum of prediction errors in anomalous window as",Sum_error_i_Cycle)
        print("Anomalous window no ", i)
        
        
print('Total Patterns detected ', Cycle_count)



#---------------------------------------------------------------------------
##PART B - POINT ANOMALY DETECTION MODULE
#---------------------------------------------------------------------------      

#---------------------------------------------------------------------------
# Step1 - Guassian distribution is assumed for training prediction errors
# Create distribution parameters mean and sd using mle method
 #---------------------------------------------------------------------------       
        
result_train['errors']


# Show distribution plot of training loss
sns.distplot(result_train['errors'], bins=50, kde=True);
plt.show()


# Use MLE to estimate best parameters for above prediction training errors
parameters = norm.fit(result_train['errors'])
print ("Mean and S.D obtained using MLE on training prediction errors is",parameters)


mean = parameters[0]      
print ("Mean",parameters[0])
stdev = parameters[1]   
print ("S.D", parameters[1])

#---------------------------------------------------------------------------
# Step 2 - Three Standard Deviation to detect point anomaly on test data
#---------------------------------------------------------------------------

Threshold = mean + 3*stdev
print(Threshold)


Point_count = list()
for i in range(len(result_test['errors'])):
    if result_test['errors'].iloc[i] > Threshold:
        Point_count.append(result_test['errors'].iloc[i])
        print('Anomalous Point Detected at Index',  "{:.4f}".format(result_test.index[i]), ":" ,"{:.2f}".format(result_test['errors'].iloc[i]))
        #print ('Point is at index', result_test.index[i])
print('Total points detected ', len(Point_count))


# Grubbs Test for outlier detection

# Grubbs.max_test_indices(data, alpha=.05)(result_test['errors'], alpha=.05)















