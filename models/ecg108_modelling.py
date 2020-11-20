# Step 1 Import libraries

import plotly.express as px
import os
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import math
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model
from tensorflow.python.keras.metrics import mse
import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = "browser"

os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

df = pd.read_csv('ecg108_test.csv')
df.describe()
df.head()
print(len(df))

# Step 2 View Dataframe and plot

# Plot 1 *** with anomalies
df_anomaly1 = df[4105:4602]  # V (PVC) and x (APC) both
df_anomaly1
df_anomaly2 = df[10876:11453]  # V (PVC) and x (APC) both
df_anomaly2

# Plot complete 60 seconds / 1 min of data with 21600 samples and 0.003 sampling frequency

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['t'], y=df['y'],
                         mode='lines',
                         name='normal'))
fig.add_trace(go.Scatter(x=df_anomaly1['t'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly1'))
fig.add_trace(go.Scatter(x=df_anomaly2['t'], y=df_anomaly2['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly2'))
fig.show()


# Step 3 Modelling dataframe - Only normal data using reconstruction concept
# Preparing data

newdf = df.drop(['Label', 'x'], axis=1)
df_model = newdf[:3961]
print(len(df_model))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_model['t'], y=df_model['y'],
                         mode='lines',
                         name='lines'))
fig.show()


# Preparing train sequences

#train_seq = df['y'].iloc[:140]

train_seq = df_model['y']
train_seq_array = (train_seq.to_numpy())

print(len(train_seq))
print(len(train_seq_array))

n = 0
step_size = 350

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

# Step 4 Model learning

## input layer
input_layer = Input(shape=(350,))

## encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

## latent view
latent_view = Dense(150, activation='sigmoid')(encode_layer3)

## decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

## output layer
output_layer = Dense(350)(decode_layer3)

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

# Step 5 Reconstruction Concept

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

# Step 6 - Qualitative Check
# Plot to see how well reconstructed data fits on actual data
mergedDf.reset_index(inplace=True)
mergedDf = mergedDf.rename(columns = {'index':'ID'})

fig = px.line(mergedDf, x='ID', y=mergedDf.columns)
fig.show()

## PART 2 - TEST DATA

# Plot to choose Test Data

# Plot 1 *** with anomalies
df_anomaly1 = df[4105:4602]  # V (PVC) and x (APC) both
df_anomaly1
df_anomaly2 = df[10876:11453]  # V (PVC) and x (APC) both
df_anomaly2

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly2['x'], y=df_anomaly2['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))

df_size = len(df)

for i in range(math.floor(df_size / 350) - 1):
    temp = i * 350
    fig.add_shape(type="line",
                  x0=350 + temp, y0=-2, x1=350 + temp, y1=2,
                  line=dict(color="Black", width=1)
                  )

fig.show()

### Test Data
newdf1 = df.drop(['Label', 'x'], axis=1)
df_test = newdf1[3175:5268]
#df_test = newdf1[0:5268]
print(len(df_model))

# Preparing test sequences

#test_seq = df['y'].iloc[140:]
test_seq = df_test['y']
test_seq_array = (test_seq.to_numpy())

print(len(test_seq))
print(len(test_seq_array))
# Need to ensure that it is taking first 400 points only

n = 0
step_size = 350

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
# Reconstruction of Test data

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
print(mergedDf_test.rename(columns={'0_x': 'reconstructed_test', '0_y': 'actual_test'}, inplace=True))


# Step 6 - Qualitative Check
# Plot to see how well reconstructed data fits on actual data
mergedDf_test.reset_index(inplace=True)
mergedDf_test = mergedDf_test.rename(columns = {'index':'ID'})

fig = px.line(mergedDf_test, x='ID', y=mergedDf_test.columns)
fig.show()

