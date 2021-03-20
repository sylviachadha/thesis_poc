# Aim - ptb-xl - deep learning ANN Classification
# Normal vs LVH
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#

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
Y_lvh = Y_short[Y_short["clean_class"] == "LVH"]
Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_lvh]
Y_short = pd.concat(frames)

# Check counts of normal and anomaly class
value_counts = Y_short['clean_class'].value_counts()
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
y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "LVH": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "LVH": 1})

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

# -----------------------------------------------------------------#
# STEP 7 - Modelling
# -----------------------------------------------------------------#
# Input ready for model as in previous step-
# X_train_short_V5 and y_train_short_num for TRAINING MODEL - 8579
# X_test_short_V5 and y_test_short_num for TESTING MODEL - 961

# Modelling # 1. Import 2. Instantiate 3. Fit 4. Predict
# --------------------------------------------------------------
# Model 1 - ANN Artificial Neural Network
# --------------------------------------------------------------
# 1. Import
import tensorflow as tf

# 2. Instantiate - Building an ANN (Architecture)
# Ann will be created as an object of class-(sequential class) which
# allows to build ann as a sequence of layers [input-hidden-output] as opposed to computational
# graph (like boltzmann m/c which r neurons connected anyway, not in
# successive layers)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))  # Shallow NN i/p n hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))  # Now deep NN hidden layer added
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # o/p layer binary so 1 neuron needed

# 3. Compile and train ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # compile
ann.fit(X_train_short_V5, y_train_short_num, batch_size=30, epochs=20)

# 4. Predict
y_pred_ann = ann.predict(X_test_short_V5)  # Predict
# Convert predicted probability to binary outcome 0 or 1
y_pred_ann1 = (y_pred_ann > 0.5)
# y_pred_ann = y_pred_ann.replace(to_replace={"False": 0, "True": 1})
# Change y_test_short_num to ndarray
y_test_short_num_arr = y_test_short_num.to_numpy()
merged_result = np.concatenate((y_pred_ann1.reshape(len(y_pred_ann1), 1), y_test_short_num_arr.reshape(len(y_test_short_num_arr), 1)), 1)
y_pred = merged_result[:, 0]

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


# Model 1 ANN

ann_result = evaluate_model(y_test_short_num_arr, y_pred)
print("ANN values for cm, accuracy, recall, precision and f1 score", ann_result)
