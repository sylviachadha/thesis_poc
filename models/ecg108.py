import plotly.express as px
import os
import pandas as pd
import plotly.graph_objects as go
import math

os.chdir('/Users/sylviachadha/Desktop/ECG_Data')

df = pd.read_csv('ecg108_test.csv')
df.describe()
df.head()
print(len(df))

# MIT-BIH Arrhythemia database
# An arrhythmia is a problem with the rate or rhythm of the heartbeat. During an arrhythmia, the heart can
# beat too fast, too slowly, or with an irregular rhythm (skipped beat or added beat)

# Plot 1 *** with anomalies
df_anomaly1 = df[4105:4602]  # V (PVC) and x (APC) both
df_anomaly1
df_anomaly2 = df[10876:11453]  # V (PVC) and x (APC) both
df_anomaly2

# Plot complete 60 seconds / 1 min of data with 21600 samples and 0.003 sampling frequency

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['t'], y=df['y'],
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly1['t'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly2['t'], y=df_anomaly2['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.show()

# Akash Thesis
# #isolate anomaly1 df['col1'][3160:5270].plot()      ##
# #isolate anomaly 2 df['col1'][9110:11750].plot()    ##
# #isolate end sequence df['col1'][18030:].plot()     ##
# Time steps for Pattern/Cycle = 370


# Plot 2
df_anomaly3 = df[3925:4198]  # V (PVC) and x (APC) both
df_anomaly3
df_anomaly4 = df[10697:11001]  # V (PVC) and x (APC) both
df_anomaly4

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['t'], y=df['y'],
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly3['t'], y=df_anomaly3['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly4['t'], y=df_anomaly4['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.show()

# Plot 3
# Same plot with samples on x-axis instead of time

# Plot 1 *** with anomalies
df_anomaly1 = df[4105:4602]  # V (PVC) and x (APC) both
df_anomaly1
df_anomaly2 = df[10876:11453]  # V (PVC) and x (APC) both
df_anomaly2

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly1['x'], y=df_anomaly1['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))
fig.add_trace(go.Scatter(x=df_anomaly2['x'], y=df_anomaly2['y'],
                         line=dict(color='red'),
                         mode='lines',
                         name='lines'))

df_size = len(df)

for i in range(math.floor(df_size / 350) - 1):
    temp = i * 350
    fig.add_shape(type="line",
                  x0=350 + temp, y0=-2, x1=350 + temp, y1=2,
                  line=dict(color="Black", width=1)
                  )

fig.show()

# As per Correlogram 1 cycle = 350 samples or 0.97 sec
