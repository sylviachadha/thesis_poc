import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


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
# These leads are (I, II, III, AVL, AVR, AVF, V1, V2, V3, V4, V5, V6)
# Signals are (0,1,2,3,4,5,6,7,8,9,10,11)
# X_train(19634,1000,12) rows are 1000 col 12
# To show record for 1 patient

# Function for 12 Lead ecg


def plot12lead(ecg_id):
    # Make a df out of ndarray
    X_df_p1 = pd.DataFrame(ecg_id)
    X_df_p1
    # Naming index of X_df_p1
    X_df_p1.index
    X_df_p1.index.name = 'Sample-number'
    # Naming columns of X_df_p1
    X_df_p1.columns = ['SignalI', 'SignalII', 'SignalIII',
                       'SignalAVL', 'SignalAVR', 'SignalAVF',
                       'SignalV1', 'SignalV2', 'SignalV3',
                       'SignalV4', 'SignalV5', 'SignalV6']
    # Reset Index
    X_df_p1.reset_index(inplace=True)

    # Dynamic Plot plotly

    df_long = pd.melt(X_df_p1, id_vars=['Sample-number'], value_vars=['SignalI', 'SignalII', 'SignalIII',
                                                                      'SignalAVL', 'SignalAVR', 'SignalAVF',
                                                                      'SignalV1', 'SignalV2', 'SignalV3',
                                                                      'SignalV4', 'SignalV5', 'SignalV6'])

    fig = px.line(df_long, x='Sample-number', y='value', color='variable',
                  title="12 Lead ECG")

    return fig.show()


# Call Function
plot12lead(X_train[0])
plot12lead(X_train[7])
plot12lead(X_train[163])
plot12lead(X_train[854])
plot12lead(X_train[153])





