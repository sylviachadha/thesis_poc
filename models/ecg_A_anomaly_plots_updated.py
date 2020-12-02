# Step 1 Import libraries

import os
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


import plotly.io as pio
pio.renderers.default = "browser"

# Step 2 Read Dataset / Custom Functions

os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

# A Read Input
def read_dataset(filename):
    df = pd.read_csv(filename)
    df.describe()
    df.head()
    print(len(df))
    return df


# B Read Annotation file
def read_annotated(filname):
    df_annotated = pd.read_csv(filname)
    df_annotated.describe()
    df_annotated.head()
    print(len(df_annotated))
    return df_annotated



# C Plot Raw data plot
def plot_PAC_anomaly(df,df_anomaly,name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name=name))
    fig.add_trace(go.Scatter(x=df_anomaly['x'], y=df_anomaly['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly-PAC'))

    return fig


# D Plot with annotated step size
def plot_PAC_anomaly_actualstep(df, df_anomaly,name):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name=name))
    fig.add_trace(go.Scatter(x=df_anomaly['x'], y=df_anomaly['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly-PAC'))

    for i in df_annotated['sample'].values:
        print(i)
        fig.add_shape(type="line",
                  x0=i, y0=-2, x1=i, y1=2,
                  line=dict(color="Black", width=1)
                  )

    return fig

#############################################################
# 1. Check another ecg record with A type of anomaly ecg 100

df = read_dataset('ecg100_A.csv')
df_annotated = read_annotated('ecg100_annotations.csv')

# Anomaly df
df_anomaly = df[2044:2402]
df_anomaly

# Call Plotting Functions
figure1 = plot_PAC_anomaly(df,df_anomaly,'normal-ecgrecord(100)')
figure1.show()

figure3 = plot_PAC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(100)')
figure3.show()


#############################################################
# 2. Check another ecg record with A type of anomaly ecg 209

df = read_dataset('ecg209_A.csv')

df_annotated = read_annotated('ecg209_annotations.csv')

# Anomaly df
df_anomaly = df[19148:19404]
df_anomaly

# Call Plotting Functions
figure1 = plot_PAC_anomaly(df,df_anomaly,'normal-ecgrecord(209)')
figure1.show()

figure3 = plot_PAC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(209)')
figure3.show()


#############################################################
# 3. Check another ecg record with V type of anomaly ecg 220

df = read_dataset('ecg220_A.csv')
df_annotated = read_annotated('ecg220_annotations.csv')


# Anomaly df
df_anomaly = df[17936:18288]
df_anomaly


# Call Plotting Functions
figure1 = plot_PAC_anomaly(df,df_anomaly,'normal-ecgrecord(220)')
figure1.show()

figure3 = plot_PAC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(220)')
figure3.show()