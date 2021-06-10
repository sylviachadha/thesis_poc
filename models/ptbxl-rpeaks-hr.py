import plotly.graph_objects as go
from biosppy.signals import ecg
import wfdb
import pandas as pd
import plotly.express as px


normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/03000/03697_lr'
record = wfdb.rdrecord(normal_file, channels=[11])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

heart_rate = out[6]
r_peaks = out[2]

X_aVF_df = pd.DataFrame(record.p_signal.flatten())
X_aVF_df.columns =['Value']

# Find values corresponding to R peak timestamp
r_value = []
for r in r_peaks:
    r_peak_value = X_aVF_df.loc[r].at["Value"]
    r_value.append(r_peak_value)


fig = px.line(X_aVF_df)
fig.add_trace(go.Scatter(x=r_peaks, y=r_value, mode="markers"))
fig.update_traces(marker=dict(size=16))
fig.show()
