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

# Step 2 Read Dataset

os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

def read_dataset(filename):
    df = pd.read_csv(filename)
    df.describe()
    df.head()
    print(len(df))
    return df

df = read_dataset('ecg219_V.csv')

# Anomaly df
df_anomaly = df[13863:14141]
df_anomaly

# Read Annotation file
def read_annotated(filname):
    df_annotated = pd.read_csv(filname)
    df_annotated.describe()
    df_annotated.head()
    print(len(df_annotated))
    return df_annotated

df_annotated = read_annotated('ecg219_annotations.csv')

# Step 3 Correlogram to find the step size # Step size 273
# To check pattern
def get_correlogram_step_size(df):
    df_filter = df[:7000]

    ## To decide no. of lags and how many largest values - if 5000 lags?
    values = acf(df_filter['y'],nlags=1500)
    correlation_values = np.round(values, 2)
    print(len(correlation_values))
    type(correlation_values)

    plot_acf(df_filter['y'], lags=1500)
    plt.show()

    # Largest values in array
    largest_n_values = heapq.nlargest(80, correlation_values)
    print(largest_n_values)
    # Check Period
    max_correlated = heapq.nlargest(80, range(len(correlation_values)), correlation_values.take)
    #max_correlated_index = sorted(max_correlated)
    print(max_correlated)  ## Appx 350 points
    return max_correlated


# Step 4 Custom Functions

# Function1 - Plot using plotly library

# A. Raw data plot
def plot_PVC_anomaly(df,df_anomaly,name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name=name))
    fig.add_trace(go.Scatter(x=df_anomaly['x'], y=df_anomaly['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly-PVC'))

    return fig

# B. Plot with chosen step size
def plot_PVC_anomaly_selectedstep(df, df_anomaly, name, step_size):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                             mode='lines',
                             name=name))
    fig.add_trace(go.Scatter(x=df_anomaly['x'], y=df_anomaly['y'],
                             line=dict(color='red'),
                             mode='lines',
                             name='anomaly-PVC'))


    df_size = len(df)

    for i in range(math.floor(df_size / step_size) - 1):
        temp = i * step_size
        fig.add_shape(type="line",
                  x0=step_size + temp, y0=-2, x1=step_size + temp, y1=2,
                  line=dict(color="Black", width=1)
                  )

    return fig

# C. Plot with annotated step size
def plot_PVC_anomaly_actualstep(df, df_anomaly,name):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name=name))
    fig.add_trace(go.Scatter(x=df_anomaly['x'], y=df_anomaly['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly-PVC'))

    for i in df_annotated['sample'].values:
        print(i)
        fig.add_shape(type="line",
                  x0=i, y0=-2, x1=i, y1=2,
                  line=dict(color="Black", width=1)
                  )

    return fig

# Call Plotting Functions
figure1 = plot_PVC_anomaly(df,df_anomaly,'normal-ecgrecord(219)')
figure1.show()

correlated_values = get_correlogram_step_size(df)
correlated_values
figure2 = plot_PVC_anomaly_selectedstep(df,df_anomaly,'normal-myannotation(219)',287)
figure2.show()

figure3 = plot_PVC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(219)')
figure3.show()


#############################################################
# 2. Check another ecg record with V type of anomaly ecg 223

df = read_dataset('ecg223_V.csv')
# Anomaly df
df_anomaly = df[8490:8776]
df_anomaly

df_annotated = read_annotated('ecg223_annotations.csv')

# Call Plotting Functions
figure1 = plot_PVC_anomaly(df,df_anomaly,'normal-ecgrecord(223)')
figure1.show()

# Get step size n plot
correlated_values = get_correlogram_step_size(df)
correlated_values
# Decide step size based on correlated_values to pass to step_size
figure2 = plot_PVC_anomaly_selectedstep(df,df_anomaly,'normal-myannotation(223)',267)
figure2.show()

figure3 = plot_PVC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(223)')
figure3.show()


#############################################################
# 3. Check another ecg record with V type of anomaly ecg 223

df = read_dataset('ecg202_V.csv')
# Anomaly df
df_anomaly = df[10346:10900]
df_anomaly

df_annotated = read_annotated('ecg202_annotations.csv')

# Call Plotting Functions
figure1 = plot_PVC_anomaly(df,df_anomaly,'normal-ecgrecord(202)')
figure1.show()

# Get step size n plot
correlated_values = get_correlogram_step_size(df)
correlated_values
# Decide step size based on correlated_values to pass to step_size
figure2 = plot_PVC_anomaly_selectedstep(df,df_anomaly,'normal-myannotation(202)',396)
figure2.show()

figure3 = plot_PVC_anomaly_actualstep(df,df_anomaly,'normal-docannotation(202)')
figure3.show()