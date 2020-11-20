## Pattern Assumption and use code of Autoencoder to model

# Akash Thesis ECG Timestep as 370 and lookback taken as 14
# Apply autoencoder concept

# ---------------------------------------------------------------------------

# Step 1 Import Libraries

# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from keras.layers import Input, Dense
from keras.models import Model

# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

# ---------------------------------------------------------------------------

# Step 2 Load dataset

# ---------------------------------------------------------------------------


df = pd.read_csv('ecg_mit.csv', header=None, names=['col0', 'y', 'col2'])
df.drop('col0', axis=1, inplace=True)
df.drop('col2', axis=1, inplace=True)
df.describe()
df.head()
len(df)

population = df['y'].plot(title="Population")
plt.show()

step_size = 370

df_anomaly1 = df[3160:5270]
df_anomaly2 = df[9110:11750]
df_test = pd.concat([df_anomaly1, df_anomaly2])
df_validation = df[16710:]

df_not_train = pd.concat([df_validation, df_test])
df_train = df[~df.isin(df_not_train)].dropna()

df_train['y'].plot(title="Training")
plt.show()


mergedDf2 = df_train['y'][0:2500]
mergedDf2.plot()
plt.show()

df_test['y'].plot(title="Testing")
plt.show()

# mergedDf.plot()
# plt.show()

# View first 1000 points only
# mergedDf2 = mergedDf[2500:3500]
# mergedDf2.plot()
# plt.show()

#  Split into train and test dataframe

# df_train, df_test = np.split(df, [int(0.7 * len(df))])

# Length of dataframes
df_train_len = len(df_train)
df_test_len = len(df_test)

print(df_train_len)
print(df_test_len)

# ---------------------------------------------------------------------------

# Step 3 Customised Function

# ---------------------------------------------------------------------------


# 3A - Plot raw data

def plot_sine_wave(y, legend):
    ax = y.plot(#figsize=(10, 8),
                title='Ecg Wave', label=legend)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    ax.set(xlabel=xlabel, ylabel=ylabel)
    xlabel = 'time-sec'
    ylabel = 'amplitude'
    plt.axhline(y=0, color='k')
    return plt

# 3B - Preparing train sequences

#train_seq = df['y'].iloc[:140]

train_seq = df_train['y']
train_seq_array = (train_seq.to_numpy())

print(len(train_seq))
print(len(train_seq_array))

n = 0
step_size = 370

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
step_size = 370

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

# View first 1000 points only
df2 = df.head(1000)
zz = plot_sine_wave(df2['y'], "")
plt.show()

# ---------------------------------------------------------------------------

# Step 4 Autoencoder Neural N/W Learning
# Just use df2 for viewing the fit after training on whole dataframe

# ---------------------------------------------------------------------------

# 1 shape has 370 data points so when we feed the sequence of 370 data points
# each data point will be treated as a column, so 370 columns will be there.


## input layer
input_layer = Input(shape=(370,))

## encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

## latent view
latent_view = Dense(100, activation='sigmoid')(encode_layer3)

## decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

## output layer
output_layer = Dense(370)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# Model Compilation

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

nb_epoch = 50
learning_rate = 0.1

ae_nn = model.fit(train_X, train_X,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)


# ---------------------------------------------------------------------------

# Step 5 Neural N/W reconstruction on only Normal Training data

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
print(mergedDf.rename(columns={'0_x': 'reconstructed_train', '0_y': 'actual_train'}, inplace=True))

# ---------------------------------------------------------------------------
# Step 5A - Qualitative Check
# Plot to see how well reconstructed data fits on actual data
# ---------------------------------------------------------------------------

#from matplotlib.pyplot import *

#mergedDf.index = df_train['t']

mergedDf.plot()
plt.show()

# View first 1000 points only
mergedDf2 = mergedDf.head(1000)
mergedDf2.plot()
plt.show()


##(((((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))))
##### Method 2
##### Time Series Concept


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from keras.layers import Input, Dense
from keras.models import Model

# Check working directory
wd = os.getcwd()
print(wd)
os.chdir('/Users/sylviachadha/Desktop/Github/Data')

# ---------------------------------------------------------------------------

# Step 2 Load dataset

# ---------------------------------------------------------------------------


df = pd.read_csv('ecg_mit.csv', header=None, names=['col0', 'y', 'col2'])
df.drop('col0', axis=1, inplace=True)
df.drop('col2', axis=1, inplace=True)
df.describe()
df.head()
len(df)

population = df['y'].plot(title="Population")
plt.show()

step_size = 370

df_anomaly1 = df[3160:5270]
df_anomaly2 = df[9110:11750]
df_test = pd.concat([df_anomaly1, df_anomaly2])
df_validation = df[16710:]

df_not_train = pd.concat([df_validation, df_test])
df_train = df[~df.isin(df_not_train)].dropna()

df_train['y'].plot(title="Training")
plt.show()

df_test['y'].plot(title="Testing")
plt.show()


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
    ax = y.plot(#figsize=(12, 6),
                title='ecg wave', label=legend)
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
n_steps = 370  ## Assumption for 1 pattern

# Split into samples
train_X, train_Y = split_sequence(train_seq, n_steps)

print(train_X.shape)

# test_seq #-------------------------------
test_seq = df_test['y'].to_list()

# choose a number of time steps
n_steps = 370  ## Assumption for 1 pattern

# Split into samples
test_X, test_Y = split_sequence(test_seq, n_steps)

print(test_X.shape)

# This shape is already in form expected by model [samples,features] so no need to reshape


#  Visualize data with anomaly
zz = plot_sine_wave(df['y'], "")

# View first 1000 points only
df2 = df.head(1000)
zz = plot_sine_wave(df2['y'], "")
plt.show()

# Define model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
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
#df_train_new = df_train.iloc[n_steps:]

#mergedDf_train.index = df_train_new['t']

# mergedDf.reset_index(inplace=True)
# mergedDf.index += 20


mergedDf_train.plot()
plt.show()

# View first 1000 points only
mergedDf2 = mergedDf_train.head(1000)
mergedDf2.plot()
plt.show()

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
#df_test_new = df_test.iloc[n_steps:]

#mergedDf.index = df_test_new['t']

mergedDf.plot()
plt.show()

# View first 1000 points only
mergedDf2 = mergedDf[2500:3500]
mergedDf2.plot()
plt.show()