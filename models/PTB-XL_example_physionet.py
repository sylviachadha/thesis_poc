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

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# 12 lead ecg
# X_train(19634,1000,12) rows are 1000 col 12
# To show record for 1 patient

# 1st Patient record - Labelled Normal
X_df_p1 = pd.DataFrame(X_train[4])
X_df_p1
X_df_p1.head(10)

print(X_df_p1.shape)

X_df_p1.plot()
plt.show()

# Extract the 3rd column as lead V2
X_df_p1_Lead2 = X_df_p1.iloc[:, 3]
type(X_df_p1_Lead2)  # Now like a series, univariate data.

# Change series to a dataframe
df_p1_V2 = pd.DataFrame({'Sample': X_df_p1_Lead2.index, 'V2 value': X_df_p1_Lead2.values})
df_p1_V2

# -------------------------------
# Interactive Plots using plotly
# --------------------------------

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_p1_V2['Sample'], y=df_p1_V2['V2 value'],
                         mode='lines',
                         name='normal'
                         ))
# fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
#                          line=dict(color='red'),
#                          mode='lines',
#                          name='anomaly_PAC'))
fig.show()

# 7th Patient record - Labelled Abnormal - MI #7,132

X_df_p7 = pd.DataFrame(X_train[132])
X_df_p7
X_df_p7.head(10)

print(X_df_p7.shape)

X_df_p7.plot()
plt.show()  # From plot green & orange lead shows abnormality V1 & V2

# Extract the 3rd column as lead V2
X_df_p7_Lead2 = X_df_p7.iloc[:, 3]
type(X_df_p7_Lead2)  # Now like a series, univariate data.

# Change series to a dataframe
df_p7_V2 = pd.DataFrame({'Sample': X_df_p7_Lead2.index, 'V2 value': X_df_p7_Lead2.values})
df_p7_V2

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_p7_V2['Sample'], y=df_p7_V2['V2 value'],
                         mode='lines',
                         name='normal'))
fig.show()
