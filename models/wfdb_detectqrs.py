# wfdb functions -  detect qrs function
# ---------------------------------------------#

# Step 1 Import libraries
#----------------------------
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

# Step 2 Import file
#-----------------------
# Define file path u want to use - normal(00001) and mi file (00008)
normal_file = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/records100/00000/00008_lr'
mi_file = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/records100/00000/00008_lr'


# Step 3 Read header and record
#-------------------------------
header = wfdb.rdheader(normal_file)
record = wfdb.rdrecord(normal_file, channels=[7])

# Step 4 Start index from 1 and plot the record after converting to df
#----------------------------------------------------------------------
df = pd.DataFrame(record.p_signal, columns=['V2_Signal_mV'])
df.index = df.index + 1
df['Time_ms'] = range(1, 10000, 10)

df.head()
df.tail()

fig = px.line(df, x='Time_ms', y="V2_Signal_mV")
fig.show()

# Step 5  to apply wfdb package's qrs detection algorithm
#----------------------------------------------------------
record.p_signal_1D =  np.ndarray.flatten(record.p_signal)

# XQRS - configuration class that stores initial parameters for detection
# detect = wfdb.processing.XQRS(record.p_signal_1D,fs=record.fs)
# This xqrs_detect function returns-
#    The indices of the detected QRS complexes.

#detect = processing.XQRS(record.p_signal[:, 0], fs=record.fs)
detect = wfdb.processing.xqrs_detect(record.p_signal_1D,fs=record.fs)

# Multiply detect indices (sample points) by 10 to plot in millisec
# x values
detect_x = detect*10

# Find corresponding y values
# y values
detect_y = df.iloc[detect].V2_Signal_mV

# detect_y is series need to change to df and then just use y values out of it
# for plotting
detect_df_y = pd.DataFrame({'index': detect_y.index, 'y_values': detect_y.values})

# Plot 1 - Soft peaks
detect_df = pd.DataFrame()
detect_df['X'] = detect_x
detect_df['Y'] = detect_df_y['y_values']

# Add traces of peaks on top of normal df

fig = px.line(df, x='Time_ms', y="V2_Signal_mV", title="ecg")
fig.add_trace(go.Scatter(x=detect_df["X"], y=detect_df["Y"], mode='markers', name='qrs detection'))

fig.show()