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
normal_file = '/Users/sylviachadha/Desktop/All_datasets/PTB-XL/records100/00000/00001_lr'
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


# Detector 0 wfdb **************
#record.p_signal_1D =  np.ndarray.flatten(record.p_signal)

#r_peaks = wfdb.processing.xqrs_detect(record.p_signal_1D,fs=record.fs)


# https://pypi.org/project/py-ecg-detectors/
# Detectors
from ecgdetectors import Detectors
detectors = Detectors(100)

# Detector 1 - Hamilton
#r_peaks = detectors.hamilton_detector(record.p_signal)

# Detector 2 - Christov working **********
#r_peaks = detectors.christov_detector(record.p_signal)

# Detector 3 - Engelse and Zeelenberg
#r_peaks = detectors.engzee_detector(record.p_signal)

# Detector 4 - Pan and Tompkins
#r_peaks = detectors.pan_tompkins_detector(record.p_signal)

# Detector 5 - Stationary Wavelet Transform
#r_peaks = detectors.swt_detector(record.p_signal)

# Detector 6 - Two Moving Average ***********
r_peaks = detectors.two_average_detector(record.p_signal)


# Common processing for all detectors
# Multiply detect indices (sample points) by 10 to plot in millisec
# x values
r_peaks_x  = [i * 10 for i in r_peaks]

# Find corresponding y values
# y values
detect_y = df.iloc[r_peaks].V2_Signal_mV

# detect_y is series need to change to df and then just use y values out of it
# for plotting
detect_df_y = pd.DataFrame({'index': detect_y.index, 'y_values': detect_y.values})

# Plot 1 - QRS detection
detect_df = pd.DataFrame()
detect_df['X'] = r_peaks_x
detect_df['Y'] = detect_df_y['y_values']

# Add traces of peaks on top of normal df

fig = px.line(df, x='Time_ms', y="V2_Signal_mV", title="ecg")
fig.add_trace(go.Scatter(x=detect_df["X"], y=detect_df["Y"], mode='markers', name='qrs detection'))

fig.show()

#### R peak detection algorithm
# Out of 7 R peak detection algorithms 3 worked.
# 1 gave wrong results, 2nd gave correct result but missed some peaks.
# last one gave correct results with all detected peaks.

# From R peak we can detect RR Interval (heart rate) as one feature.



