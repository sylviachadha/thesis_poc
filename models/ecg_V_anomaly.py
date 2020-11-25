#---------------------------------------------------------------------------------------
# Plots for ecg records/files having PVC arrhythmia [Premature Ventricular Contraction]
# 3 datasets - # 202, #219 and #223  all with PVC anomaly type
#---------------------------------------------------------------------------------------

# Step 1 - Import libraries
import plotly.express as px
import os
import pandas as pd
import plotly.graph_objects as go
import math
import plotly.io as pio
pio.renderers.default = "browser"

# Step 2 Custom Functions

# Function1 - Choose Dataset to Plot
def get_dataset(dataset):
    os.chdir('/Users/sylviachadha/Desktop/ECG_Data')
    df = pd.read_csv(dataset)
    df.describe()
    df.head()
    print(len(df))
    return df

# Function2 - Plot using plotly library

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


# Step 3 - Plots with anomalies
# Files for Type V [ Premature ventricular contraction] anomaly are 202, 219 and 223

# Plot 1 *** ecg file 202 ML11 Signal
df1 = get_dataset('ecg202_V.csv')
df_anomaly1 = df1[10346:10900]
df_anomaly1
figure1 = plot_PVC_anomaly(df1,df_anomaly1,'normal-ecgrecord(202)')
figure1.show()


# Plot 2 *** ecg file 219 ML11 Signal
df2 = get_dataset('ecg219_V.csv')
df_anomaly2 = df2[13863:14141]
df_anomaly2
figure2 = plot_PVC_anomaly(df2,df_anomaly2,'normal-ecgrecord(219)')
figure2.show()


# Plot 3 *** ecg file 223 ML11 Signal
df3 = get_dataset('ecg223_V.csv')
df_anomaly3 = df3[8490:8776]
df_anomaly3
figure3 = plot_PVC_anomaly(df3,df_anomaly3,'normal-ecgrecord(223)')
figure3.show()

