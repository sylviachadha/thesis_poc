# Classification in Lead V5 and V6
# Diseases LVH (Hypertrophy Class) and CLBBB (Conduction Disturbance Class)

# Step 1 Import libraries
#----------------------------
import ast

import plotly.express as px
import plotly.io as pio
import pandas as pd
import wfdb
import plotly.graph_objects as go
import numpy as np
import math
import matplotlib.pyplot as plt
from wfdb import processing
pio.renderers.default = "browser"

# Load agregated df and make ecg-id as index column
path = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/'
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')


# Load scp_statements.csv annotations
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

#?
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Apply diagnostic subclass

def aggregate_diagnostic1(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)

    return list(set(tmp))
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)


# Removing brackets and only keeping the text
Y['stripped_class'] = Y['diagnostic_subclass'].apply(' '.join)

norm_df = Y[Y['stripped_class'] == 'NORM']
#clbbb_df = Y[Y['stripped_class'] == 'CLBBB'] ## IMI greatest
lvh_df = Y[Y['stripped_class'] == 'LVH']

#norm_df = norm_df.head(500) # If reduce normal also to same like LVH 500

# Preparing Y df with only Norm, CLBBB and LVH labels
#Y = norm_df.append(clbbb_df, ignore_index=True)
Y = norm_df.append(lvh_df, ignore_index=True)
#Y = Y.append(lvh_df, ignore_index=True)
del Y['stripped_class']


# Preparing X
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/'
sampling_rate = 100
X = load_raw_data(Y, sampling_rate, path)


# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_subclass

# Step 2
# Convert to two dimensional data X_train and X_test
# instead of 3 dimensional as expected by ML algo

df_train = pd.DataFrame()

for i in range(len(X_train)):
    v2_col = X_train[i][:, 10]
    df_temp = pd.DataFrame(v2_col.reshape(-1, len(v2_col)))
    df_train = df_train.append(df_temp)


df_test = pd.DataFrame()

for i in range(len(X_test)):
    v2_col = X_test[i][:, 10]
    df_temp = pd.DataFrame(v2_col.reshape(-1, len(v2_col)))
    df_test = df_test.append(df_temp)

# Training the Decision Tree Classification model on the Training set
y_train = y_train.apply(' '.join)
y_test = y_test.apply(' '.join)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(df_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(df_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.values.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

