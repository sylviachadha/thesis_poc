# Problem
# Sequence/Pattern Anomaly Detection using Neural Networks
# Reconstruction concept, detecting unusual shapes using Autoencoder

# Problem Use Case/Significance
# Unusual condition corresponds to an unusual shape in medical conditions

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model
import matplotlib.pyplot as plt
#import plotly.io as pio
#pio.renderers.default = "browser"
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------#
# Step 1 - Draw 1000 normal pattern
# --------------------------------------------------------------#

fi = 0.5  # Inherent freq 1 so 1 cycle time is 1/1 = 1 sec so 10 sec = 10 cycles
t = 10
fs = 100  # Sampling freq 100 so sampling time interval is 0.01
sample_points = 1000
ts = 0.01
a = 2

# 1000 Signal values as a function of 1000 time values
time = np.arange(0, 10, 0.01)

# Plot 1 sine pattern
sig1_sine = a * np.sin(2 * np.pi * fi * time)
# Create normal plots also with different a and f values?

# Introduce randomness to data
noise = np.random.normal(0, .1, sig1_sine.shape)
new_sig1_sine = sig1_sine + noise
print(new_sig1_sine)
sine_pattern = new_sig1_sine.copy()

# Create 1000 patterns

for n in range(999):
    new_row = a * np.sin(2 * np.pi * fi * time)
    noise1 = np.random.normal(0, .1, new_row.shape)
    new_pattern = new_row + noise1
    sine_pattern = np.vstack([sine_pattern, new_pattern])

# Plot this pattern (Change ndarray to df to plot)
# Change ndarray to df
df = pd.DataFrame(sine_pattern)

# Single Plot
df.iloc[150].plot()
plt.show()

# --------------------------------------------------------------#
# Step 2 - Draw 1000 abnormal random pattern
# --------------------------------------------------------------#

df_a = df.copy()

tot_rows = len(df_a.index)
print(tot_rows)

anomalydf = pd.DataFrame()
for index, row in df_a.iterrows():
        print(row)
        r = random.randint(0, 850) # picking location of anomaly
        print(r)
        index_list = [range(r, r + 100, 1)] # anomaly of 100 points
        print(index_list)
        distort_f_index = r
        print(distort_f_index)
        distort_l_index = r+100
        print(distort_l_index)
        n = 50 # out of 100 points anomaly split into 50,50 so looks closer to real both will make df_distort2
        r = random.randint(-3, 3)
        r1 = random.randint(-3, 3)

        x1 = [r] * n
        x2 = [r1] * n
        x1_arr = np.array(x1)
        x2_arr = np.array(x2)

        noise1 = np.random.normal(0, .1, x1_arr.shape)
        noise2 = np.random.normal(0, .1, x2_arr.shape)

        new_x1 = x1_arr + noise1
        new_x2 = x2_arr + noise2

        df_arr1 = pd.DataFrame(new_x1, columns=['Signal'])
        df_arr2 = pd.DataFrame(new_x2, columns=['Signal'])
        frame = [df_arr1, df_arr2]
        df_distort2 = pd.concat(frame)
        print(df_distort2)
        df_distort2.index = np.arange(start=distort_f_index, stop=distort_l_index, step=1)

        dfn1 = row.iloc[0:distort_f_index].to_frame(name="Signal")
        dfn3 = row.iloc[distort_l_index:].to_frame(name="Signal")

        # Concatenate dfn1 + df_distort1 + dfn3
        frames = [dfn1, df_distort2, dfn3]
        new_row = pd.concat(frames)
        new_row = new_row.transpose()
        anomalydf = anomalydf.append(new_row, ignore_index=True)


# Single Plot
anomalydf.iloc[128].plot()
plt.show()


# ------------------------------
# Change df to ndarray
# ------------------------------
df_arr = df.to_numpy() # all normal patterns
anomalydf_arr = anomalydf.to_numpy() # all abnormal patterns

# Concatenate total data (normal + abnormal)
X_2D = np.concatenate((df_arr, anomalydf_arr))


# -------------------------------------------
# See plots to verify 2D Input data
# -------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and triangle from
# index 50 to 99
X_2D_df1 = X_2D_df.iloc[765]
plt.plot(X_2D_df1)
plt.show()

# -------------------------------------------
# STEP 3 Data to input to ML Algorithms
# X_2D and y_1D
# --------------------------------------------

# X_2D as created above
# y is label
y_class0 = np.zeros(1000)
y_class1 = np.ones(1000)
y = np.concatenate((y_class0, y_class1))
y_1D = y


# -------------------------------------------------------------------
# STEP 4 - Split into Train & Test to feed to ML Algorithm
# ----------------------------------------------------------
# Check Shape of data and target/label
print("Shape of data", X_2D.shape)
print("Shape of label", y_1D.shape)

# Split into Train and Test
# The stratify parameter asks whether you want to retain the same proportion
# of classes in the train and test sets that are found in the entire
# original dataset
X_train, X_test, y_train, y_test = train_test_split(X_2D, y_1D, test_size=0.2, stratify=y) #random_state=4)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

# Check number of anomalies in y_train and y_test

print("Train", Counter(y_train))
print("Test", Counter(y_test))

# ------------------------------------------------------------------#
# STEP 5 - Feature Scaling, compulsory for deep learning, scale
# everything all features.
# fit & transform on train but only transform on test
# -----------------------------------------------------------------#

# ----------------------------------------------------------------------------#
# STEP 6 - One CLass methods / Unsupervised methods preprocessing
# Steps - a. Define architecture-> b. compile-> c. fit model on training data
# ----------------------------------------------------------------------------#
# Require only normal data for training but require both normal &
# anomalous data for test

X_train_1class = X_train[y_train == 0]


# Autoencoder Model Architecture
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

nb_epoch = 30
learning_rate = 0.1

# c. Model fit on training data (X,X as reconstruction concept)
ae_nn = model.fit(X_train_1class, X_train_1class,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)

# ----------------------------------------------------------------------------#
# Step 7 AE prediction on Training data (which is only normal data)
# ---------------------------------------------------------------------------#

# Predicted and actual arrays, need to flatten both because both are sequences of length 1000
pred = model.predict(X_train_1class)

pred_train = pred.flatten()

actual_train = X_train_1class.flatten()

print(len(pred_train))
print(len(actual_train))

# Change to df
predicted_df_train = pd.DataFrame(pred_train)
actual_df_train = pd.DataFrame(actual_train)

# Merge two dataframes based on index

mergedDf = predicted_df_train.merge(actual_df_train, left_index=True, right_index=True)
print(len(mergedDf))
# mergedDf_test
print(mergedDf.columns)
print(mergedDf.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# Just take 1 record to test that is record 0 out of all records
# print(mergedDf.rename(columns={'0_x': 'predicted_train', '0_y': 'actual_train'}, inplace=True))

# Plot Single Train
mergedDf1 = mergedDf.head(1000)
mergedDf1.plot()
plt.show()

# Get threshold value from training
my_list_train = []
n = 0
while n < len(mergedDf):
    mergedDf_train_new = mergedDf[n:n+1000]
    mse_train_row = mean_squared_error(mergedDf_train_new['actual_train'],mergedDf_train_new['predicted_train'])
    my_list_train.append(mse_train_row)
    n = n+1000

import statistics
# mean error
avg_train_mean_error = statistics.mean(my_list_train)
avg_train_mean_error = round(avg_train_mean_error,2)
print('average training mse', avg_train_mean_error)
# stdev around mean
train_stdev_mean = statistics.stdev(my_list_train, xbar=avg_train_mean_error)
print('stdev around mean error', train_stdev_mean)
three_stdev_train = train_stdev_mean*3

# threshold = mean + 3 sd
threshold_val = avg_train_mean_error + three_stdev_train
threshold_val = round(threshold_val,5)
print(threshold_val)

# ----------------------------------------------------------------------------#
# Step 8 AE prediction on Test data (which is both normal and abnormal data)
# ---------------------------------------------------------------------------#

# Predicted and actual arrays, need to flatten both because both are sequences of length 1000
pred1 = model.predict(X_test)
pred_test = pred1.flatten()

actual_test = X_test.flatten()

print(len(pred_test))
print(len(actual_test))

# Change predicted and actual arrays to dataframe to see the plot
# ---------------------------------------------------------------------------
predicted_df_test = pd.DataFrame(pred_test)
actual_df_test = pd.DataFrame(actual_test)

# Merge two dataframes based on index

mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
print(len(mergedDf_test))
# mergedDf_test
print(mergedDf_test.columns)
print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))

# To see individual plot

# mergedDf_test has 2 columns pred and actual and 20K points i.e. 20 patterns
# Filter 1 record (First 3 records are normal)
mergedDf_test1 = mergedDf_test.head(1000)
mergedDf_test1.plot()
plt.show()

mergedDf_test1['actual_test'].plot()
plt.show()

mergedDf_test1['predicted_test'].plot()
plt.show()

# 2nd record is abnormal in test
mergedDf_test2 = mergedDf_test[2000:3000]
mergedDf_test2.plot()
plt.show()

mergedDf_test2['actual_test'].plot()
plt.show()

mergedDf_test2['predicted_test'].plot()
plt.show()

# --------------------------------------------------#
# Step 9 Quantitative method to check anomaly
# --------------------------------------------------#

# PART A - INDIVIDUAL POINTS

# PART B - GROUP (PATTERN WISE OF 10 SEC/1000 DATA POINTS)
# ----------------------------------------------------------#
# Group results of 400 patterns - pred and actual (since we need to decide label
# based on the group)
# df of length 400 rows (i.e 400 patterns only)
mergedDf_test_grp = mergedDf_test.copy()
N = 1000

# Calculate mse for every 1000 points which makes up a single pattern
# and there are 400 patterns in test so we will get 400 mse error values

# split dataframe by row
my_list_test = []
n = 0
while n < len(mergedDf_test_grp):
    mergedDf_test_new = mergedDf_test_grp[n:n+1000]
    mse_row = mean_squared_error(mergedDf_test_new['actual_test'],mergedDf_test_new['predicted_test'])
    mse_row = round(mse_row, 5)
    my_list_test.append(mse_row)
    n = n+1000

# Change list to df
list_df = pd.DataFrame (my_list_test, columns=['mse'])
list_df['actual_label'] = y_test

# Threshold to declare anomaly from training is threshold_val

list_df['pred_label'] = 0
for index, x in enumerate(list_df['mse']):
    if x > threshold_val:
        list_df['pred_label'][index] = 1


# Evaluation Metrics
def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return (cm, acc, recall, precision, f1score)


ae_result = evaluate_model(list_df['actual_label'], list_df['pred_label'])

def print_results():
    cm = confusion_matrix(list_df['actual_label'], list_df['pred_label'])
    tn, fp, fn, tp = cm.ravel()
    print("anomaly as anomaly tp = ", tp)
    print("anomaly as normal fn = ", fn)
    print("normal as normal tn = ", tn)
    print("normal as anomaly fp = ", fp)
    print("accuracy = ", ae_result[1])
    print("recall = ", ae_result[2])
    print("precision = ", ae_result[3])
    print("f1-score = ", ae_result[4])
    return()

print_results()

# --------------------------------------#
# Step 10 Error visualization
# --------------------------------------#

# Plot histogram of errors
list_df['mse'].hist()
plt.show()

# Plot Q-Q Plot
from statsmodels.graphics.gofplots import qqplot
qqplot(list_df['mse'], line='s')
plt.show()

