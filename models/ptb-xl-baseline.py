# Method C for Segment w.r.t P-QRS-T (Before and After R Peak)
# -----------------------------------------------------------------
# DETECTING BASELINE AFTER SEGMENTATION
# -----------------------------------------------------------------

# Should be okay for 1 patient but will look different for all patients
# Cardiac cycle function from BioSppy package

# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (21831) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/00000/00043_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()


# Method 1
# ---------------------------------------------------------------------
# Get maximum point of individual cycles

cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
                                                rpeaks=r_peaks,
                                                sampling_rate=record.fs,
                                                before=0.30,
                                                after=0.50)

# Plot cardiac cycles obtained as ndarray with shape (14,65)

import numpy as np

f = np.array(cardiac_cycles)

f.shape
no_of_seq = f.shape[0]
print('no of seq', no_of_seq)


i = 0
while i < no_of_seq:
    print(i)
    seq = f[i, :]
    #print(seq)
    seq_series = pd.Series(seq)
    freq = seq_series.value_counts()
    freq_reset = freq.reset_index()
    freq_new = freq_reset.rename(columns={'index': 'value', 0: 'freq'})
    value = freq_new.iloc[0]
    print(value)
    hor_line_value = freq_new['value'].iloc[0]
    plt.plot(seq)
    plt.axhline(y=hor_line_value, color='g', linestyle='-')
    plt.show()
    i = i + 1

# # Dummy example to pick most frequently occuring number
# import pandas as pd
# import numpy as np
# data = np.array([1,5,5,5,6,7,8,8,5])
# s = pd.Series(data)
# print(s)
# s.value_counts()

# Method 2
# ---------------------------------------------------------------------
# To get the baseline on 10 seconds signal, get the point with
# maximum occurence to check
# record.p_signal is ndarray, change to dataframe

record_df = pd.DataFrame(record.p_signal, columns = ['signal-value'])
# With value counts 1000 points reduced to 418 points
freq = record_df.value_counts()
freq_res = freq.reset_index()
freq_n = freq_res.rename(columns={'index': 'value', 0: 'freq'})
hor_line = freq_n['signal-value'].iloc[0]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.axhline(y=hor_line, color='b', linestyle='-')
plt.show()

# Draw individual cycles at 0.16

cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
                                                rpeaks=r_peaks,
                                                sampling_rate=record.fs,
                                                before=0.30,
                                                after=0.50)

# Plot cardiac cycles obtained as ndarray with shape (14,65)

import numpy as np

f = np.array(cardiac_cycles)

f.shape
no_of_seq = f.shape[0]
print('no of seq', no_of_seq)


i = 0
while i < no_of_seq:
    print(i)
    seq = f[i, :]
    #print(seq)
    seq_series = pd.Series(seq)
    hor_line_value = hor_line
    plt.plot(seq)
    plt.axhline(y=hor_line_value, color='g', linestyle='-')
    plt.show()
    i = i + 1


# Method 3
# ---------------------------------------------------------------------
# Get 40 ms signal value before R wave for baseline (general)
# However 50 ms works better in our experiments

# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (21831) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/00000/00042_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

# Signal values corresponding to r peaks


# Each sample point represents 0.01 seconds (1000*.01 = 10 seconds)
# 1st R peak at 81 so 81*.01 = 0.81 sec
# To get baseline subtract 40 ms or .04 seconds from r peak and get signal
# value at that point to mark the baseline

cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
                                                rpeaks=r_peaks,
                                                sampling_rate=record.fs,
                                                before=0.30,
                                                after=0.50)

# Plot cardiac cycles obtained as ndarray with shape (14,65)

import numpy as np

f = np.array(cardiac_cycles)

f.shape
no_of_seq = f.shape[0]
print('no of seq', no_of_seq)


i = 0
while i < no_of_seq:
    print(i)
    seq = f[i, :]
    #print(seq) R peak at 30 in every seq
    # Signal at 26 timestamp will be baseline
    seq_series = pd.Series(seq)
    # freq = seq_series.value_counts()
    # freq_reset = freq.reset_index()
    # freq_new = freq_reset.rename(columns={'index': 'value', 0: 'freq'})
    # value = freq_new.iloc[0]
    # print(value)
    #hor_line_value = freq_new['value'].iloc[0]
    hor_line_value = seq_series[25]
    plt.plot(seq)
    plt.axhline(y=hor_line_value, color='g', linestyle='-')
    plt.show()
    i = i + 1


