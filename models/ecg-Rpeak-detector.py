import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# -----------------------------------------------------------------#
# STEP 1 - Load Data (As per paper)
# -----------------------------------------------------------------#

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

# ------------------------------------------------------------------#
# STEP 2 - Extracting Class of Anomaly (AMI) out of 71 classes in Y
# AMI Place of injury - Anterior surface so look in V1 to V4
# Reciprocal changes in lead 11, 111 and avF
# ------------------------------------------------------------------#
# Once original X and y are loaded u shorten the y to NORMAL
# and ANOMALY class u want and then load corresponding X again


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

# Get unique value counts from a column of a dataframe to
# choose which abnormal class has the most data
uniqueValues = Y_short['clean_class'].unique()
uniqueValues_count = Y_short['clean_class'].nunique()
from collections import Counter
all_unique_values = Counter(Y_short['clean_class'])
print(all_unique_values)


# Make a short df with anomaly and normal class
Y_anomaly = Y_short[Y_short["clean_class"] == "IMI"]
Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_anomaly]
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
test_fold = 10
# Train
X_train = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass

# -----------------------------------------------------------------#
# STEP 4 - Plot 12 lead ecg
# -----------------------------------------------------------------#


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
plot12lead(X_train[0]) # Normal Train
plot12lead(X_train[9280])   # IMI Train


# -----------------------------------------------------------------#
# STEP 5 - Find R peaks on the raw ecg signal
# -----------------------------------------------------------------#

from ecgdetectors import Detectors
detectors = Detectors(100)

# Change to 1 Lead ecg before detecting R peaks

# Choose Signal II
X_train_s = X_train[:, :, 1]
X_test_s = X_test[:, :, 1]

# Change ndarray to dataframe
X_train_s = pd.DataFrame(X_train_s)

norm_ecg = X_train_s.iloc[10]
mi_ecg = X_train_s.iloc[9280]

# Change series to df

df1 = pd.DataFrame(data=norm_ecg.index, columns=['timestamp'])
df2 = pd.DataFrame(data=norm_ecg.values, columns=['value'])
df = pd.merge(df1, df2, left_index=True, right_index=True)

# import pandas as pd
# pd.options.plotting.backend = "plotly"
# # using Plotly Express via the Pandas backend
# fig1 = norm_ecg.plot.line()
# plt.scatter(anomalies['timestamp'], anomalies['value'],c='r',label='Point anomalies')
# #fig2 = mi_ecg.plot.line()
# fig2.show()

# 3. Pan and Tompkins
r_peaks_pt = detectors.pan_tompkins_detector(norm_ecg)
print(r_peaks_pt)

#Location of R peaks

# for peak in r_peaks_pt:
#     df.loc[df.timestamp == i]
#     print(i)


# peak_time_value_pair = df.loc[df.timestamp == 31]
# print(peak_time_value_pair)


#f = plt.figure()

plt.title("normal ecg with R peaks ")
plt.xlabel('Time',fontsize=14)
plt.ylabel('Value',fontsize=14)
plt.plot(df['timestamp'],df['value'], '-gD', markevery=r_peaks_pt, mfc='r')
# plt.scatter(peak_time_value_pair['timestamp'], peak_time_value_pair['value'],c='r',label='Point anomalies')

plt.legend()
plt.show()


# ALL ECG R PEAK DETECTORS
# 1. Hamilton Detector
# r_peaks = detectors.hamilton_detector(mi_ecg)
# print(r_peaks)


# 2. Christov Detector
# r_peaks_ch = detectors.christov_detector(norm_ecg)
# print(r_peaks_ch)


# # 3. Pan and Tompkins
# r_peaks_pt = detectors.pan_tompkins_detector(norm_ecg)
# print(r_peaks_pt)
#
# #Location of R peaks
#
# peak_time_value_pair = df.loc[df.timestamp == 32]
# print(peak_time_value_pair)


# 4. Stationary Wavelet Transform
# r_peaks = detectors.swt_detector(mi_ecg)
# print(r_peaks)

# 5. Two moving average
# r_peaks = detectors.two_average_detector(mi_ecg)
# print(r_peaks)

# Plot R peaks on top of Line ecg chart

