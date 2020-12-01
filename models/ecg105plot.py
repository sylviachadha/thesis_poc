# ECG 105 WITH all anomalies

import wfdb
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "browser"


def set_color(y):
    if y == "N":
        return "blue"
    elif y == "A":
        return "crimson"
    elif y == "V":
        return "red"
    elif y == "~":
        return "black"
    elif y == "|":
        return "mediumturquoise"
    elif y == "+":
        return "olivedrab"


record = wfdb.rdrecord('data/105')
df_input = pd.DataFrame()
df_input['y'] = record.p_signal[:, 0]

df = df_input[:108001]

annotation = wfdb.rdann('data/105', 'atr')
df_label = pd.DataFrame()
df_label['sample'] = annotation.sample[:]
df_label['label'] = annotation.symbol[:]
df_label = df_label.iloc[:426]
print("5 min data annotations", df_label['label'].value_counts())

fig = go.Figure()

fig.add_trace(go.Scattergl(x=df.index, y=df['y'],
                           mode="lines",
                           name="normal",
                           hoverinfo='skip'
                           ))

start_sample = df_label.iloc[0]['sample']
start_label = df_label.iloc[0]['label']

i = 0

# iterrows is function of pandas df, returns 2 things so
# give 2 things, _ and row
# iterrows returns index and row and then iterate on row
for _, row in df_label.iterrows():
    if i == 0:
        i = i + 1
        continue
    end_sample = row['sample']
    end_label = row['label']

    if start_label != 'N':
        fig.add_trace(go.Scattergl(x=df.iloc[start_sample:end_sample].index, y=df.iloc[start_sample:end_sample]['y'],
                                   marker=dict(color=set_color(start_label)),
                                   name=start_label,
                                   text=start_label))
    start_label = end_label
    start_sample = end_sample

for i in df_label['sample'].values:
    fig.add_shape(type="line",
                  x0=i, y0=-2, x1=i, y1=2,
                  line=dict(color="black", width=2)
                  )

fig.show()
