# wfdb functions -  find_peaks function
# ---------------------------------------------#

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

# Define file path u want to use - normal(00001) and mi file (00008)
normal_file = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/records100/00000/00001_lr'
mi_file = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/records100/00000/00008_lr'

# After running below check record object, it has sig_name V2 and units mV
# This header and record is for full file 10 seconds / 10*1000=10K milliseconds
# Can also extract peaks for just 1000 ms data which is 1 second
header = wfdb.rdheader(normal_file)
record = wfdb.rdrecord(normal_file, channels=[7])

# The record object has signal contained in p_signal as ndarray and we
# want to plot this ndarray
# To plot we first convert ndarrray to dataframe

df = pd.DataFrame(record.p_signal, columns=['V2_Signal_mV'])
df.index = df.index + 1

# Now we have 0 to 999 sample data points (1000) in X-axis and we know this is 10 seconds of
# data or 10*1000 = 10,000 milliseconds of data

# start 0ms  stop 10,000ms  -  need 1000 data points in between

# range x-axis as 1 to 10,000 milliseconds with step size 10 so we get
# 1000 data points in between for X-axis since y-axis has 1000 points

# 1-1000 sample is 10 seconds of data or 10*1000=10K millisec
# 10 seconds or 10000 milliseconds =  1000 samples
# 1 millisec = 1000/10000 = 1/10th sample
# 10 millisec = 1 sample so every reading should be at interval of 10 millisec

df['Time_ms'] = range(1, 10000, 10)

df.head()
df.tail()

# Plot df with 2 columns created as V2_Signal value-mV and Time-milliseconds

fig = px.line(df, x='Time_ms', y="V2_Signal_mV")
fig.show()

# Apply function (find_peaks) of wfdb processing library
hard_peaks, soft_peaks = wfdb.processing.find_peaks(record.p_signal[:, 0])
# record.p_signal[:, 0] just takes all values of signal same as df['V2_Signal value-mV']
# just the format difference

# Hard peaks and Soft peaks returned is ndarray and values as per sample number 1-1000
# as per x-axis so we need to change x coordinates in milliseconds & find their
# corresponding y values inorder to plot the peaks
# We need to plot these peaks so we change ndarray to df first


# 1-1000 sample is 10 seconds of data or 10*1000=10K millisec
# 10 seconds or 10000 milliseconds =  1000 samples
# 1 millisec = 1000/10000 = 1/10th sample
# 10 millisec = 1 sample
# Since all samples multiple by 10 to get values in ms so soft &
# hard peaks also to multiple by 10 to get x axis location in millisec

# x value in millisec
s_peaks_millisec = soft_peaks * 10
h_peaks_millisec = hard_peaks * 10

# y value in mV
yy_soft = df.iloc[soft_peaks].V2_Signal_mV
yy_hard = df.iloc[hard_peaks].V2_Signal_mV

# Change series to df and name columns of df

yy_soft_df = pd.DataFrame({'index': yy_soft.index, 'y_values': yy_soft.values})
yy_hard_df1 = pd.DataFrame({'index': yy_hard.index, 'y_values': yy_hard.values})

# Plot 1 - Soft peaks
scatter_df_soft = pd.DataFrame()
scatter_df_soft['X'] = s_peaks_millisec
scatter_df_soft['Y'] = yy_soft_df['y_values']

# Plot 2 - Hard peaks
scatter_df_hard = pd.DataFrame()
scatter_df_hard['X'] = h_peaks_millisec
scatter_df_hard['Y'] = yy_hard_df1['y_values']

# Add traces of peaks on top of normal df
# fig = px.line(px.line(df, x='Time_ms', y="V2_Signal_mV",
#                       labels=dict(X="Time-ms", Y="V2_Signal-mV",
#                                   title='NORM Ecg plot - Lead:' + record.sig_name[0])))

fig = px.line(df, x='Time_ms', y="V2_Signal_mV", title="ecg")
fig.add_trace(go.Scatter(x=scatter_df_soft["X"], y=scatter_df_soft["Y"], mode='markers', name='Soft peaks'))
fig.add_trace(go.Scatter(x=scatter_df_hard["X"], y=scatter_df_hard["Y"], mode='markers', name='Hard peaks'))

fig.show()
