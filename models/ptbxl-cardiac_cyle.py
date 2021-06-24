# Cardiac Cycle Visualization after segmentation
# -------------------------------------------------

# A) Segment w.r.t RR - more accurate but difficult to interpret for further feature extraction

# B) Segment w.r.t PQRST (Your 6 second rule output) - (easier to interpret for
# further feature extraction, start of cycle ? isoelectric line or P wave)

# C) Segment based on before and After R peak rule
# (easier to interpret for further feature extraction)

# Method A

# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (21831) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/21000/21831_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()

# Change ndarray to df
record.p_signal_df = pd.DataFrame(record.p_signal)
# Convert ndarray to list
rpeaks_tolist = r_peaks.tolist()

# Since all sequences are variable length so cannot stack

import numpy as np
array_tuple = []
# Create sequences based on r peaks
rpeak_len = len(rpeaks_tolist)

seq_df = pd.DataFrame()
r = 0
while r < (rpeak_len-1):
    subsequence = record.p_signal_df[rpeaks_tolist[r]:rpeaks_tolist[r+1]]
    print(subsequence)
    #subsequence = subsequence.transpose()
    subsequence.plot()
    plt.show()
    r = r+1
    array_tuple.append(subsequence)

# Possible cons
# 1. All cycles are of different length
# 2. Not understandable as per P-QRS-T Cardiac cycle pattern

# -----------------------------------------------------------------

# Method B Segment w.r.t P-QRS-T (Your 6 second rule output)

# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (21831) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/21000/21831_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()

r_peaks_df = pd.DataFrame(r_peaks)
hr_r_peaks = r_peaks_df[(r_peaks_df[0] >= 200) & (r_peaks_df[0] <= 800)]
tot_r_peaks = len(hr_r_peaks)
hr = tot_r_peaks*10
print('Heart-rate/bpm', hr)

# Calculate fixed length cardiac cycle from heart rate
c_cycle_duration = 60/hr
print(c_cycle_duration)

# Plot cardiac cycles individually for 0.75 sec each
# Convert duration to sample points
cycle_step = c_cycle_duration * 100
print(cycle_step)


no_of_cycles = round(record.sig_len/cycle_step,0)
print(no_of_cycles)

# Change ndarray to df
record.p_signal_df = pd.DataFrame(record.p_signal)
# Convert ndarray to list
#vrpeaks_tolist = r_peaks.tolist()
cycle_points_lst = list(range(0, 1000, 75))

# Now all sequences are of same length

array_tuple = []
# Create sequences based on cycle points
cycle_points_len = len(cycle_points_lst)

seq_df = pd.DataFrame()
r = 0
while r < (cycle_points_len-1):
    subsequence = record.p_signal_df[cycle_points_lst[r]:cycle_points_lst[r+1]]
    print(subsequence)
    subsequence.plot()
    plt.show()
    r = r+1
    array_tuple.append(subsequence)


# Method C Segment w.r.t P-QRS-T (Before and After R Peak)
# -----------------------------------------------------------------

# Should be okay for 1 patient but will look different for all patients
# Cardiac cycle function from BioSppy package

# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (21831) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/21000/21831_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()

cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
                                                rpeaks=r_peaks,
                                                sampling_rate=record.fs,
                                                before=0.2,
                                                after=0.45)

# Plot cardiac cycles obtained as ndarray with shape (14,65)

import numpy as np
f = np.array(cardiac_cycles)

f.shape
no_of_seq = f.shape[0]
print('no of seq',no_of_seq )

i=0
while i < no_of_seq:
    print(i)
    seq = f[i,:]
    plt.plot(seq)
    plt.show()
    i=i+1


