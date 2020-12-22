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

# Step 2

# Add labels to X (12 signals) - Diagnostic superclass and subclass
# for 21837 patients

v2_list = []

for i in range(len(X)):
    v2_col = X[i][:, 7]
    v2_row = v2_col.reshape(-1, len(v2_col))
    v2_list.append(v2_row)

df1 = pd.DataFrame(np.concatenate(v2_list))

# Copy superclass and Subclass to this signals sheet (X) which will be
# used as an input to ml model

# df1 index starts from 0 however Y["diagnostic_superclass"]
# index starts from 1 so we first changed df1 index to
# start from 1 cz it copy based on index
df1.index = np.arange(1,len(df1)+1)

selected_columns = Y["diagnostic_superclass"]
df1['diagnostic_superclass'] = selected_columns.copy()

selected_columns = Y["diagnostic_subclass"]
df1['diagnostic_subclass'] = selected_columns.copy()

# Step 3 # Pending todo

# Handle Missing Data - to confirm (depends on % of missing data)





