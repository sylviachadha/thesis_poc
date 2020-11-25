# Step 1 - Import libraries
import plotly.express as px
import os
import pandas as pd
import plotly.graph_objects as go
import math
import plotly.io as pio
pio.renderers.default = "browser"

# Step 2 Import Dataset
os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

df = pd.read_csv('ecg100_A.csv')
df.describe()
df.head()
print(len(df))

# Step 3 - Plots with anomalies
# Files for Type A [ Premature atrial contraction] anomaly are
# Files 100, 209 and 220


# Plot 1 *** ecg file 100 with Type A anomaly as per annotation
df_anomaly1 = df[2044:2402]
df_anomaly1

# Plot complete 60 seconds / 1 min of data with 21600 samples and 0.003 sampling frequency

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name='normal'))
fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly_PAC'))
fig.show()


# Plot 2 *** ecg file 209 with Type A anomaly as per annotation
df_anomaly1 = df[19148:19404]
df_anomaly1

# Plot complete 60 seconds / 1 min of data with 21600 samples and 0.003 sampling frequency

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name='normal'))
fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly_PAC'))
fig.show()


# Plot 3 *** ecg file 220 with Type A anomaly as per annotation
df_anomaly1 = df[17936:18288]
df_anomaly1

# Plot complete 60 seconds / 1 min of data with 21600 samples and 0.003 sampling frequency

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name='normal'))
fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='anomaly_PAC'))
fig.show()