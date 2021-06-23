# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# Random Check for Normal (109,21831, 5298) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/05000/05298_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
# heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()

# cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
#                                                 rpeaks=r_peaks,
#                                                 sampling_rate=record.fs,
#                                                 before=0.2,
#                                                 after=0.4)


# Heart Rate Output

# Method 1 - Using BioSPPY Algorithm
# -------------------------------------------------------------------
heart_rate = out[6]

# Method 2 - Using Rule 1
# -------------------------------------------------------------------

# http://www.meddean.luc.edu/lumen/meded/medicine/skills/ekg/les1prnt.htm
# The basic way to calculate the rate is quite simple. You take the duration between
# two identical points of consecutive EKG waveforms such as the R-R duration.
# Take this duration and divide it into 60. The resulting equation would be:
# Rate = 60/(R-R interval)


# Calculate RR interval
# Change ndarray to df
r_peaks_df = pd.DataFrame(r_peaks)
df_len = len(r_peaks_df)
hrate_list = []
rr_interval_list = []
for r in r_peaks_df:
    while r < (df_len-1):
        print(r)
        lower_bound = r_peaks_df.iloc[r]
        upper_bound = r_peaks_df.iloc[r+1]
        # 1 sample point = 0.01 seconds
        rr_interval = ((upper_bound - lower_bound)*0.01)
        rr_interval_list.append(rr_interval)
        hrate = 60/rr_interval
        print(hrate)
        r = r+1
        hrate_list.append(hrate)


rr_interval_df = pd.DataFrame(rr_interval_list)
rr_interval_transpose = rr_interval_df.transpose()

hrate_df = pd.DataFrame(hrate_list)
hrate_df_transpose = hrate_df.transpose()

# Method 3 - Using Rule 2
# -------------------------------------------------------------------

# Count the number of RR intervals between two Tick marks (6 seconds) in the rhythm
# strip and multiply by 10 to get the bpm. This method is more effective when the
# rhythm is irregular. - # In our case we take b/w 2 sec and 8 sec duration

# R peaks lie between 200 and 800
# Gives Approximate heart rate and can be used for both normal and abnormal rhythms
hr_r_peaks = r_peaks_df[(r_peaks_df[0] >= 200) & (r_peaks_df[0] <= 800)]
tot_r_peaks = len(hr_r_peaks)
hr = tot_r_peaks*10
print('Heart-rate/bpm', hr)



