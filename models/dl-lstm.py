# Deep Learning - Long Short Term Memory
# -------------------------------------------------

# Stacked LSTM
# Part 1 Data Preprocessing
# Part 2 Building the RNN
# Part 3 Making predictions & visualizing the results

# Part 1 Data Preprocessing
# 1a import libraries
# numpy for making arrays which are only allowed inputs of neural n/w's
# as opposed to dataframes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1b Import the training set
os.chdir('D:\\Study\\All_datasets\\rnn_dataset')
dataset_train = pd.read_csv('google_stock_price_train.csv')  # df
training_set = dataset_train.iloc[:, 1:2] # df with only column open
training_set = dataset_train.iloc[:, 1:2].values # creates a numpy array
# this is 1 column numpy array as expected by nn and not a vector

# 1c Feature scaling to optimize the training process
# Standardization and Normalization 2 best ways of applying feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# 1d Reshaping the data
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    # RNN is memorizing what happened in previous 60 timesteps
    # Sliding window of size 60 sliding with stride of 1
    # to predict next value
    y_train.append(training_set_scaled[i, 0])
    # lists to numpy arrays so to make acceptable by nn

X_train,y_train = np.array(X_train), np.array(y_train)

#  To make compatible with input shape of RNN
# What u want to rehape? X_train
# No of observations: X_train.shape[0] i.e. 1198 rows
# Timesteps: X_train.shape[1] i.e. previous 60 timesteps
# Indicators / predictors / features: 1 i.e only open price

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Same in ecg here 1198 observations, 60 timesteps and 1 feature which is only
# open price so series of 60 made from the open price itself.
# In ecg 21K observations, 1000 timesteps and 1 feature which is only 1 lead
# lets say lead V6 and 1000 timesteps are made from that lead itself.

# Part 2 Building the RNN (Stacked LSTM)
# -------------------------------------------------

# Dropout regularization to avoid overfitting

# 2a Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# 2b Iniitialize RNN as a sequence of layers as opposed to the computational graph

regressor = Sequential()  # We call it regressor as opposed to classifier
# here because we are predicting continuous value so we r doing some regression
# Regression - predicting a continuous value
# Classification - predicting a category / class

# 2c Add first LSTM layer & add some Dropout Regularization

regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))  # 20% dropout i.e 10 neurons will be ignored/dropped out during each iteration of training

# regressor is an object of Sequential class. Sequential class contains the
# add method and regressor object of Sequential class can use this method.

# units is LSTM cells or no. of memory cells u want to have in this layer
# return_sequences = True in stacked LSTM once done with LSTM layers False
# input_shape in 3D corresponding to observations, timesteps and the indicators.
# observations taken automatically so just specify timesteps and features in
# in the input shape.

# 2d Add second LSTM layer with some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

# 2e Add third LSTM layer with some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

# 2f Add fourth LSTM layer with some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# 2g Add the output layer
regressor.add(Dense(units = 1))

# 2h Compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
# In classification we generally use loss = 'binary_crossentropy'
# In regression generally loss = 'mse'

# 2i Fitting the RNN to training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
# Increase epochs until there is a convergence of the loss
# Starts from 0.09 and i.e 9% loss in 1st epoch and converges at .0015 at the end
# If u get loss too small maybe e-10 something it, indicates overfitting
# When u do training on training set u must be careful not obtain overfitting
# and not try to decrease loss as much as possible.

# Part 3 Making predictions and visualizing the results

# 3a Get real stock price
dataset_test = pd.read_csv('google_stock_price_test.csv')  # df)
real_stock_price = dataset_test.iloc[:, 1:2].values

# 3b Get predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# vertical concatenation axis = 0
# horizontal concatenation axis = 10
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
# Special 3d structure expected by predict method of NN

X_test = []
for i in range(60, 80):# Test data contains 20 financial days
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
# Reshape to 3D structure as expected by NN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
# To inverse the scaling of our predictions because our regressor was trained
# to predict the scaled values of stock prices. But we should never change the
# actual test values so to get the original scale of scaled predicted values
# we apply inverse transform method from our scaling sc object.
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# 3c Visualizing the result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Our model lags behind in cases where there are spikes because it cannot react to fast non linear changes.
# The spike is a fast non linear change so our model cannot react properly
# Its okay as our model reacts good to smooth changes and manages to follow
# upward and downward trends.