# Problem # FULL PATTERN OF 1000 POINTS
# ############################################################################
# Sequence/Pattern Anomaly Detection using Neural Networks
# Reconstruction concept, detecting unusual shapes using Autoencoder

# Problem Use Case/Significance
# Unusual condition corresponds to an unusual shape in medical conditions

# Aim - ptb-xl - # Normal vs Abnormal
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import Counter
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/'
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

# ---------------------------------------------------------------#
# STEP 2 - Extracting Class of Anomaly out of 71 classes in Y
# ---------------------------------------------------------------#
# Once original X and y are loaded u shorten the y to NORMAL
# and ANOMALY class u want and then load corresponding X again

# Add clean class
Y_short = Y.copy()
Y_short['clean_class'] = Y_short['diagnostic_subclass'].apply(' '.join)


# Make a short df with anomaly and normal class
# Y_anomaly = Y_short[Y_short["clean_class"] == "LVH"]
Y_anomaly = Y_short[Y_short["clean_class"] != "NORM"]


#Y_short['Anomaly'] = ["NORM" if x == "NORM" else "Anomaly" for x in Y_short['clean_class']]

Y_anomaly['diagnostic_subclass'] = Y_anomaly['clean_class'].apply(lambda x: 'Anomaly')


Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_anomaly]
Y_short = pd.concat(frames)

# Check counts of normal and anomaly class
value_counts = Y_short['diagnostic_subclass'].value_counts()
print(value_counts)

# Since Filtering and value counts done, remove the clean class
del Y_short['clean_class']

# Load corresponding X as per Y_short
X_short = load_raw_data(Y_short, sampling_rate, path)

# -----------------------------------------------------------------#
# STEP 3 - Train/Test Split   (# AS PAPER)
# -----------------------------------------------------------------#
# 10 fold stratified sampling
test_fold = 10
# Train
X_train_short = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train_short = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test_short = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test_short = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass

# -----------------------------------------------------------------#
# STEP 4 - Extracting Channel / Lead out of 12 leads
# Preparing X for the model
# -----------------------------------------------------------------#
# Change X from 3D to 2D with only Lead V5 Signal for each patient

print(X_train_short.shape)
print(X_test_short.shape)
# Select only Lead V5 from X which is column 10
# Leads (I, II, III, AVL, AVR, AVF, V1, ..., V6)

X_train_short_V5 = X_train_short[:, :, 10]
X_test_short_V5 = X_test_short[:, :, 10]

# Before feed to model need to change y_train_short & y_test_short
# labels from ['NORM'] to NORM

# -----------------------------------------------------------------#
# STEP 5 - Changing ['NORM'] to NORM and further to 0 (encoding)
# Preparing y for the model
# -----------------------------------------------------------------#

# Change ['NORM'] to NORM - since applying to series no need to specify
# column names
y_train_short.index.name
y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

# Replace NORM with 0 and Anomaly Class with 1
y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "A n o m a l y": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "A n o m a l y": 1})


trainlabel_counts = y_train_short_num.value_counts()
print(trainlabel_counts)

# print shapes of new X objects
print("X_train shape", X_train_short_V5.shape)
print("X_test shape", X_test_short_V5.shape)

# print shapes of new y objects
print("y_train shape", y_train_short_num.shape)
print("y_test shape", y_test_short_num.shape)

# Check number of anomalies in y_train and y_test

print("Train", Counter(y_train_short_num))
print("Test", Counter(y_test_short_num))


# ------------------------------------------------------------------#
# STEP 5 - Feature Scaling, compulsory for deep learning, scale
# everything all features.
# fit & transform on train but only transform on test
# -----------------------------------------------------------------#
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train_short_V5 = sc.fit_transform(X_train_short_V5)
# X_test_short_V5 = sc.transform(X_test_short_V5)


# ----------------------------------------------------------------------------#
# STEP 6 - One CLass methods / Unsupervised methods preprocessing
# Steps - a. Define architecture-> b. compile-> c. fit model on training data
# ----------------------------------------------------------------------------#
# Require only normal data for training but require both normal &
# anomalous data for test

X_train_1class = X_train_short_V5[y_train_short_num == 0]
#X_train_1class_short = X_train_1class[:, 0:1000]


# Autoencoder Model Architecture
# input layer
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model

input_layer = Input(shape=(1000,))

# encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

# latent view
latent_view = Dense(50, activation='sigmoid')(encode_layer3)

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

nb_epoch = 5
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
# mergedDf1.plot()
# plt.show()

mergedDf1['actual_train'].plot()
plt.show()
mergedDf1['predicted_train'].plot()
plt.show()

mergedDf2 = mergedDf[6000:7000]
# mergedDf2.plot()
# plt.show()

mergedDf2['actual_train'].plot()
plt.show()
mergedDf2['predicted_train'].plot()
plt.show()


mergedDf3 = mergedDf[2752000:2753000]
# mergedDf2.plot()
# plt.show()

mergedDf3['actual_train'].plot()
plt.show()
mergedDf3['predicted_train'].plot()
plt.show()

####----------- END OF TRAINING, PATTERN LEARNT OR NOT #####---------


# from sklearn.metrics import mean_squared_error
#
# # Get threshold value from training
# my_list_train = []
# n = 0
# while n < len(mergedDf):
#     mergedDf_train_new = mergedDf[n:n+1000]
#     mse_train_row = mean_squared_error(mergedDf_train_new['actual_train'],mergedDf_train_new['predicted_train'])
#     my_list_train.append(mse_train_row)
#     n = n+1000
#
# import statistics
# # mean error
# avg_train_mean_error = statistics.mean(my_list_train)
# avg_train_mean_error = round(avg_train_mean_error,2)
# print('average training mse', avg_train_mean_error)
# # stdev around mean
# train_stdev_mean = statistics.stdev(my_list_train, xbar=avg_train_mean_error)
# print('stdev around mean error', train_stdev_mean)
# three_stdev_train = train_stdev_mean*1
#
# # threshold = mean + 3 sd
# threshold_val = avg_train_mean_error + three_stdev_train
# threshold_val = round(threshold_val,5)
# print(threshold_val)
#
# # ----------------------------------------------------------------------------#
# # Step 8 AE prediction on Test data (which is both normal and abnormal data)
# # ---------------------------------------------------------------------------#
#
# # Predicted and actual arrays, need to flatten both because both are sequences of length 1000
# pred1 = model.predict(X_test_short_V5)
# pred_test = pred1.flatten()
#
# actual_test = X_test_short_V5.flatten()
#
# print(len(pred_test))
# print(len(actual_test))
#
# # Change predicted and actual arrays to dataframe to see the plot
# # ---------------------------------------------------------------------------
# predicted_df_test = pd.DataFrame(pred_test)
# actual_df_test = pd.DataFrame(actual_test)
#
# # Merge two dataframes based on index
#
# mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
# print(len(mergedDf_test))
# # mergedDf_test
# print(mergedDf_test.columns)
# print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))
#
# # To see individual plot
#
# # mergedDf_test has 2 columns pred and actual and 20K points i.e. 20 patterns
# # Filter 1 record (First 3 records are normal)
# mergedDf_test1 = mergedDf_test.head(1000)
# mergedDf_test1.plot()
# plt.show()
#
# # Individual plot
# mergedDf_test1['predicted_test'].plot()
# plt.show()
# mergedDf_test1['actual_test'].plot()
# plt.show()
#
# # 2nd record is abnormal in test
# mergedDf_test4 = mergedDf_test[2000:3000]
# mergedDf_test4.plot()
# plt.show()
#
#
# # --------------------------------------------------#
# # Step 9 Quantitative method to check anomaly
# # --------------------------------------------------#
#
# # PART A - INDIVIDUAL POINTS
#
# # PART B - GROUP (PATTERN WISE OF 10 SEC/1000 DATA POINTS)
# # ----------------------------------------------------------#
# # Group results of 400 patterns - pred and actual (since we need to decide label
# # based on the group)
# # df of length 400 rows (i.e 400 patterns only)
# mergedDf_test_grp = mergedDf_test.copy()
# N = 1000
#
# # Calculate mse for every 1000 points which makes up a single pattern
# # and there are 400 patterns in test so we will get 400 mse error values
#
# # split dataframe by row
# my_list_test = []
# n = 0
# while n < len(mergedDf_test_grp):
#     mergedDf_test_new = mergedDf_test_grp[n:n+1000]
#     mse_row = mean_squared_error(mergedDf_test_new['actual_test'],mergedDf_test_new['predicted_test'])
#     mse_row = round(mse_row, 5)
#     my_list_test.append(mse_row)
#     n = n+1000
#
# # Change list to df
# list_df = pd.DataFrame (my_list_test, columns=['mse'])
# list_df['actual_label'] = y_test_short_num.values
#
# # Threshold to declare anomaly from training is threshold_val
#
# list_df['pred_label'] = 0
# for index, x in enumerate(list_df['mse']):
#     if x > threshold_val:
#         list_df['pred_label'][index] = 1
#
#
# # Evaluation Metrics
# def evaluate_model(y_actual, y_pred):
#     cm = confusion_matrix(y_actual, y_pred)
#     print(cm)
#     acc = accuracy_score(y_actual, y_pred)
#     recall = recall_score(y_actual, y_pred)
#     precision = precision_score(y_actual, y_pred)
#     f1score = f1_score(y_actual, y_pred)
#     return (cm, acc, recall, precision, f1score)
#
#
# ae_result = evaluate_model(list_df['actual_label'], list_df['pred_label'])
#
# def print_results():
#     cm = confusion_matrix(list_df['actual_label'], list_df['pred_label'])
#     tn, fp, fn, tp = cm.ravel()
#     print("anomaly as anomaly tp = ", tp)
#     print("anomaly as normal fn = ", fn)
#     print("normal as normal tn = ", tn)
#     print("normal as anomaly fp = ", fp)
#     print("accuracy = ", ae_result[1])
#     print("recall = ", ae_result[2])
#     print("precision = ", ae_result[3])
#     print("f1-score = ", ae_result[4])
#     return()
#
# print_results()
#
# # --------------------------------------#
# # Step 10 Error visualization
# # --------------------------------------#
#
# # Plot histogram of errors
# list_df['mse'].hist()
# plt.show()
#
# # Plot Q-Q Plot
# from statsmodels.graphics.gofplots import qqplot
# qqplot(list_df['mse'], line='s')
# plt.show()
#


##############################################################

# Problem - SHORT PATTERN OF 100 POINTS ######################
#----------------------------------------
# Sequence/Pattern Anomaly Detection using Neural Networks
# Reconstruction concept, detecting unusual shapes using Autoencoder

# Problem Use Case/Significance
# Unusual condition corresponds to an unusual shape in medical conditions

# Aim - ptb-xl - # Normal vs Abnormal
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import Counter
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/'
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

# ---------------------------------------------------------------#
# STEP 2 - Extracting Class of Anomaly out of 71 classes in Y
# ---------------------------------------------------------------#
# Once original X and y are loaded u shorten the y to NORMAL
# and ANOMALY class u want and then load corresponding X again

# Add clean class
Y_short = Y.copy()
Y_short['clean_class'] = Y_short['diagnostic_subclass'].apply(' '.join)


# Make a short df with anomaly and normal class
# Y_anomaly = Y_short[Y_short["clean_class"] == "LVH"]
Y_anomaly = Y_short[Y_short["clean_class"] != "NORM"]


#Y_short['Anomaly'] = ["NORM" if x == "NORM" else "Anomaly" for x in Y_short['clean_class']]

Y_anomaly['diagnostic_subclass'] = Y_anomaly['clean_class'].apply(lambda x: 'Anomaly')


Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_anomaly]
Y_short = pd.concat(frames)

# Check counts of normal and anomaly class
value_counts = Y_short['diagnostic_subclass'].value_counts()
print(value_counts)

# Since Filtering and value counts done, remove the clean class
del Y_short['clean_class']

# Load corresponding X as per Y_short
X_short = load_raw_data(Y_short, sampling_rate, path)

# -----------------------------------------------------------------#
# STEP 3 - Train/Test Split   (# AS PAPER)
# -----------------------------------------------------------------#
# 10 fold stratified sampling
test_fold = 10
# Train
X_train_short = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train_short = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test_short = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test_short = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass

# -----------------------------------------------------------------#
# STEP 4 - Extracting Channel / Lead out of 12 leads
# Preparing X for the model
# -----------------------------------------------------------------#
# Change X from 3D to 2D with only Lead V5 Signal for each patient

print(X_train_short.shape)
print(X_test_short.shape)
# Select only Lead V5 from X which is column 10
# Leads (I, II, III, AVL, AVR, AVF, V1, ..., V6)

X_train_short_V5 = X_train_short[:, :, 10]
X_test_short_V5 = X_test_short[:, :, 10]

# Before feed to model need to change y_train_short & y_test_short
# labels from ['NORM'] to NORM

# -----------------------------------------------------------------#
# STEP 5 - Changing ['NORM'] to NORM and further to 0 (encoding)
# Preparing y for the model
# -----------------------------------------------------------------#

# Change ['NORM'] to NORM - since applying to series no need to specify
# column names
y_train_short.index.name
y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

# Replace NORM with 0 and Anomaly Class with 1
y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "A n o m a l y": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "A n o m a l y": 1})

testlabel_counts = y_test_short_num.value_counts()
print(testlabel_counts)
trainlabel_counts = y_train_short_num.value_counts()
print(trainlabel_counts)

# print shapes of new X objects
print("X_train shape", X_train_short_V5.shape)
print("X_test shape", X_test_short_V5.shape)

# print shapes of new y objects
print("y_train shape", y_train_short_num.shape)
print("y_test shape", y_test_short_num.shape)

# Check number of anomalies in y_train and y_test

print("Train", Counter(y_train_short_num))
print("Test", Counter(y_test_short_num))


# ------------------------------------------------------------------#
# STEP 5 - Feature Scaling, compulsory for deep learning, scale
# everything all features.
# fit & transform on train but only transform on test
# -----------------------------------------------------------------#
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train_short_V5 = sc.fit_transform(X_train_short_V5)
# X_test_short_V5 = sc.transform(X_test_short_V5)


# ----------------------------------------------------------------------------#
# STEP 6 - One CLass methods / Unsupervised methods preprocessing
# Steps - a. Define architecture-> b. compile-> c. fit model on training data
# ----------------------------------------------------------------------------#
# Require only normal data for training but require both normal &
# anomalous data for test

X_train_1class = X_train_short_V5[y_train_short_num == 0]
X_train_1class_short = X_train_1class[:, 0:100]


# Autoencoder Model Architecture
# input layer
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense
from keras.models import Model, load_model

input_layer = Input(shape=(100,))

# encoding architecture
encode_layer1 = Dense(15, activation='relu')(input_layer)
encode_layer2 = Dense(10, activation='relu')(encode_layer1)
encode_layer3 = Dense(5, activation='relu')(encode_layer2)

# latent view
latent_view = Dense(5, activation='sigmoid')(encode_layer3)

# decoding architecture
decode_layer1 = Dense(15, activation='relu')(latent_view)
decode_layer2 = Dense(10, activation='relu')(decode_layer1)
decode_layer3 = Dense(5, activation='relu')(decode_layer2)

# output layer
output_layer = Dense(100)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# b. Model Compilation

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

nb_epoch = 5
learning_rate = 0.1

# c. Model fit on training data (X,X as reconstruction concept)
ae_nn = model.fit(X_train_1class_short, X_train_1class_short,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=1)

# ----------------------------------------------------------------------------#
# Step 7 AE prediction on Training data (which is only normal data)
# ---------------------------------------------------------------------------#

# Predicted and actual arrays, need to flatten both because both are sequences of length 1000
pred = model.predict(X_train_1class_short)

pred_train = pred.flatten()

actual_train = X_train_1class_short.flatten()

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
mergedDf1 = mergedDf.head(100)
# mergedDf1.plot()
# plt.show()

mergedDf1['actual_train'].plot()
plt.show()
mergedDf1['predicted_train'].plot()
plt.show()

mergedDf2 = mergedDf[600:700]
# mergedDf2.plot()
# plt.show()

mergedDf2['actual_train'].plot()
plt.show()
mergedDf2['predicted_train'].plot()
plt.show()


mergedDf3 = mergedDf[1500:1600]
# mergedDf2.plot()
# plt.show()

mergedDf3['actual_train'].plot()
plt.show()
mergedDf3['predicted_train'].plot()
plt.show()

####----------- END OF TRAINING, PATTERN LEARNT OR NOT #####---------


# from sklearn.metrics import mean_squared_error
#
# # Get threshold value from training
# my_list_train = []
# n = 0
# while n < len(mergedDf):
#     mergedDf_train_new = mergedDf[n:n+1000]
#     mse_train_row = mean_squared_error(mergedDf_train_new['actual_train'],mergedDf_train_new['predicted_train'])
#     my_list_train.append(mse_train_row)
#     n = n+1000
#
# import statistics
# # mean error
# avg_train_mean_error = statistics.mean(my_list_train)
# avg_train_mean_error = round(avg_train_mean_error,2)
# print('average training mse', avg_train_mean_error)
# # stdev around mean
# train_stdev_mean = statistics.stdev(my_list_train, xbar=avg_train_mean_error)
# print('stdev around mean error', train_stdev_mean)
# three_stdev_train = train_stdev_mean*1
#
# # threshold = mean + 3 sd
# threshold_val = avg_train_mean_error + three_stdev_train
# threshold_val = round(threshold_val,5)
# print(threshold_val)
#
# # ----------------------------------------------------------------------------#
# # Step 8 AE prediction on Test data (which is both normal and abnormal data)
# # ---------------------------------------------------------------------------#
#
# # Predicted and actual arrays, need to flatten both because both are sequences of length 1000
# pred1 = model.predict(X_test_short_V5)
# pred_test = pred1.flatten()
#
# actual_test = X_test_short_V5.flatten()
#
# print(len(pred_test))
# print(len(actual_test))
#
# # Change predicted and actual arrays to dataframe to see the plot
# # ---------------------------------------------------------------------------
# predicted_df_test = pd.DataFrame(pred_test)
# actual_df_test = pd.DataFrame(actual_test)
#
# # Merge two dataframes based on index
#
# mergedDf_test = predicted_df_test.merge(actual_df_test, left_index=True, right_index=True)
# print(len(mergedDf_test))
# # mergedDf_test
# print(mergedDf_test.columns)
# print(mergedDf_test.rename(columns={'0_x': 'predicted_test', '0_y': 'actual_test'}, inplace=True))
#
# # To see individual plot
#
# # mergedDf_test has 2 columns pred and actual and 20K points i.e. 20 patterns
# # Filter 1 record (First 3 records are normal)
# mergedDf_test1 = mergedDf_test.head(1000)
# mergedDf_test1.plot()
# plt.show()
#
# # Individual plot
# mergedDf_test1['predicted_test'].plot()
# plt.show()
# mergedDf_test1['actual_test'].plot()
# plt.show()
#
# # 2nd record is abnormal in test
# mergedDf_test4 = mergedDf_test[2000:3000]
# mergedDf_test4.plot()
# plt.show()
#
#
# # --------------------------------------------------#
# # Step 9 Quantitative method to check anomaly
# # --------------------------------------------------#
#
# # PART A - INDIVIDUAL POINTS
#
# # PART B - GROUP (PATTERN WISE OF 10 SEC/1000 DATA POINTS)
# # ----------------------------------------------------------#
# # Group results of 400 patterns - pred and actual (since we need to decide label
# # based on the group)
# # df of length 400 rows (i.e 400 patterns only)
# mergedDf_test_grp = mergedDf_test.copy()
# N = 1000
#
# # Calculate mse for every 1000 points which makes up a single pattern
# # and there are 400 patterns in test so we will get 400 mse error values
#
# # split dataframe by row
# my_list_test = []
# n = 0
# while n < len(mergedDf_test_grp):
#     mergedDf_test_new = mergedDf_test_grp[n:n+1000]
#     mse_row = mean_squared_error(mergedDf_test_new['actual_test'],mergedDf_test_new['predicted_test'])
#     mse_row = round(mse_row, 5)
#     my_list_test.append(mse_row)
#     n = n+1000
#
# # Change list to df
# list_df = pd.DataFrame (my_list_test, columns=['mse'])
# list_df['actual_label'] = y_test_short_num.values
#
# # Threshold to declare anomaly from training is threshold_val
#
# list_df['pred_label'] = 0
# for index, x in enumerate(list_df['mse']):
#     if x > threshold_val:
#         list_df['pred_label'][index] = 1
#
#
# # Evaluation Metrics
# def evaluate_model(y_actual, y_pred):
#     cm = confusion_matrix(y_actual, y_pred)
#     print(cm)
#     acc = accuracy_score(y_actual, y_pred)
#     recall = recall_score(y_actual, y_pred)
#     precision = precision_score(y_actual, y_pred)
#     f1score = f1_score(y_actual, y_pred)
#     return (cm, acc, recall, precision, f1score)
#
#
# ae_result = evaluate_model(list_df['actual_label'], list_df['pred_label'])
#
# def print_results():
#     cm = confusion_matrix(list_df['actual_label'], list_df['pred_label'])
#     tn, fp, fn, tp = cm.ravel()
#     print("anomaly as anomaly tp = ", tp)
#     print("anomaly as normal fn = ", fn)
#     print("normal as normal tn = ", tn)
#     print("normal as anomaly fp = ", fp)
#     print("accuracy = ", ae_result[1])
#     print("recall = ", ae_result[2])
#     print("precision = ", ae_result[3])
#     print("f1-score = ", ae_result[4])
#     return()
#
# print_results()
#
# # --------------------------------------#
# # Step 10 Error visualization
# # --------------------------------------#
#
# # Plot histogram of errors
# list_df['mse'].hist()
# plt.show()
#
# # Plot Q-Q Plot
# from statsmodels.graphics.gofplots import qqplot
# qqplot(list_df['mse'], line='s')
# plt.show()
#
