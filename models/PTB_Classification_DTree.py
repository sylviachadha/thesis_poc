# PTB Classification Normal vs MI
# Step 1 - Template
import pandas as pd
import numpy as np
import wfdb
import ast
import os
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylviachadha/Desktop/PTB-XL Database/'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data

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

Y['stripped_class'] = Y['diagnostic_superclass'].apply(' '.join)

norm_df = Y[Y['stripped_class'] == 'NORM']
mi_df = Y[Y['stripped_class'] == 'MI']

min_norm_df = norm_df.head(3000)

Y = min_norm_df.append(mi_df, ignore_index=True)

X = load_raw_data(Y, sampling_rate, path)


del Y['stripped_class']

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# Step 2
# Convert to two dimensional data X_train and X_test
# instead of 3 dimensional as expected by ML algo

df_train = pd.DataFrame()


for i in range(len(X_train)):
    v2_col = X_train[i][:, 7]
    df_temp = pd.DataFrame(v2_col.reshape(-1, len(v2_col)))
    df_train = df_train.append(df_temp)


df_test = pd.DataFrame()

for i in range(len(X_test)):
    v2_col = X_test[i][:, 7]
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




