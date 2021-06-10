# Aim - ptb-xl - deep learning ANN Classification
# Normal vs LVH
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#

import pandas as pd
import numpy as np
import wfdb
import ast
import plotly.express as px
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
Y = pd.read_csv(path + 'ptbxl_db_filter.csv', index_col='ecg_id')
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

# Select only combination of NORM and SR as NORM

def aggregate_diagnostic1(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)

print(Y['diagnostic_superclass'])

print(len(Y))

# -------------------------------------------------------#
# STEP 2 - Plot all Data (Only Normal and without noise
# -------------------------------------------------------#

# Select only 1 lead (avF)

# Select only Lead aVF from X which is column 5
# Leads (I, II, III, AVL, AVR, AVF, V1, ..., V6)

X_aVF = X[:, :, 6]

# ----------------------------------------------------#
# STEP 3 - Plot all 4699 records avf & put in a folder
# ----------------------------------------------------#

# Change ndarray to dataframe
X_aVF_df = pd.DataFrame(X_aVF)

# Plots
fig = px.line(X_aVF_df.iloc[0])
fig.show()

fig = px.line(X_aVF_df.iloc[100])
fig.show()







