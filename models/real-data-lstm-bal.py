# Aim - ptb-xl - deep learning LSTM Classification
# Normal vs Anomaly - all classes inorder to make dataset balanced
# -----------------------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# -----------------------------------------------------------------#

import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylviachadha/Desktop/Thesis/Datasets/ptb-xl/'
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

# -----------------------------------------------------------------#
# STEP 6 - Feature Scaling, compulsory for deep learning, scale
# everything all features
# -----------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_short_V5 = sc.fit_transform(X_train_short_V5)
X_test_short_V5 = sc.transform(X_test_short_V5)

# Reshape AS Input expected by LSTM
X_train_short_V5 = np.reshape(X_train_short_V5, (X_train_short_V5.shape[0],X_train_short_V5.shape[1], 1))

# -----------------------------------------------------------------#
# STEP 7 - Modelling (Building the RNN - Stacked LSTM)
# -----------------------------------------------------------------#
# Modelling # 1. Import 2. Instantiate 3. Fit 4. Predict
# --------------------------------------------------------------
# Model 1 - LSTM (Long Short Term Memory)
# --------------------------------------------------------------
# 1. Import
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense  # to add o/p layer
from keras.layers import LSTM   # to add lstm layer
from keras.layers import Dropout

# 2. Instantiate - Building an LSTM (Architecture)
# LSTM will be created as an object of class-(sequential class) which
# allows to build rnn as a sequence of layers [input-hidden-output] as opposed to computational
# graph (like boltzmann m/c which r neurons connected anyway, not in
# successive layers)
rnn_lstm = tf.keras.models.Sequential()
rnn_lstm.add(LSTM(units=50, return_sequences=True, input_shape = (X_train_short_V5.shape[1], 1)))
rnn_lstm.add(Dropout(0.2))
# Add 2nd LSTM layer & some Dropout Regularization
rnn_lstm.add(LSTM(units=50, return_sequences=True))
rnn_lstm.add(Dropout(0.2))
# Add 3rd LSTM layer & some Dropout Regularization
rnn_lstm.add(LSTM(units=50, return_sequences=True))
rnn_lstm.add(Dropout(0.2))
# Add 4th LSTM layer & some Dropout Regularization
rnn_lstm.add(LSTM(units=50))
rnn_lstm.add(Dropout(0.2))
# Adding the output layer
rnn_lstm.add(Dense(units=1))


# 3. Compile and train LSTM  # Optimizer RMSprop
rnn_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # compile
rnn_lstm.fit(X_train_short_V5, y_train_short_num, batch_size=1000, epochs=10)

# 4. Predict

# Before Prediction reshape test data into format expected by LSTM
X_test_short_V5 = np.reshape(X_test_short_V5, (X_test_short_V5.shape[0],X_test_short_V5.shape[1], 1))

y_pred_rnn_lstm = rnn_lstm.predict(X_test_short_V5)  # Predict
# Convert predicted probability to binary outcome 0 or 1

# Try Threshold = 0.5
#y_pred_rnn_lstm_ann2 = (y_pred_rnn_lstm > 0.5)

# Try Threshold = 0.6
y_pred_rnn_lstm_ann2 = (y_pred_rnn_lstm > 0.6)


# Change y_test_short_num to ndarray
y_test_short_num_arr1 = y_test_short_num.to_numpy()

# Merge to show pred and actual result side by side
merged_result1 = np.concatenate((y_pred_rnn_lstm_ann2.reshape(len(y_pred_rnn_lstm_ann2), 1), y_test_short_num_arr1.reshape(len(y_test_short_num_arr1), 1)), 1)
print(merged_result1)
y_pred1 = merged_result1[:, 0]


# STEP 7 - Evaluation Metrics
# ------------------------------------------------------------------

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return (cm, acc, recall, precision, f1score)


# Model 1 LSTM

rnn_lstm_result1 = evaluate_model(y_test_short_num_arr1, y_pred1)
print("LSTM values for cm, accuracy, recall, precision and f1 score", rnn_lstm_result1)
