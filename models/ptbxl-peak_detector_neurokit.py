# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk

# Step 1
# ------------------------------------------------------------------------------------
# Plot a 10 sec record with r peaks from biosppy
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/00000/00109_lr'
# These leads are (I, II, III, AVL, AVR, AVF, V1, V2, V3, V4, V5, V6)
# Signals are (0,1,2,3,4,5,6,7,8,9,10,11)
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()


# Step 2
# ------------------------------------------------------------------------------------
# Clean the ECG Signal, detect R peak and redraw
# ecg_signal can be list, np.array, pd.Series
# record.p_signal.flatten (changing 2D ndarray to 1D ndarray)

ecg = record.p_signal.flatten()
ecg_series = pd.Series(ecg)

ecg_clean_sig = nk.ecg_clean(ecg_series, sampling_rate=100, method='neurokit')

plt.plot(ecg_clean_sig)
plt.show()


# Step 3
# ------------------------------------------------------------------------------------
# Find R peaks using neurokit2 and plot them on cleaned signal

ecg_r_tuple = nk.ecg_peaks(ecg_clean_sig, sampling_rate=100, method='neurokit', correct_artifacts=False)

ecg_r_dict = ecg_r_tuple[1]  # dictionary

# Extract peak values from dictionary using key name which is ECG_R_Peaks
# value = dictionary[key]
r_peak = ecg_r_dict['ECG_R_Peaks']

ecg_sig_len = len(ecg_clean_sig)
plt.plot(range(ecg_sig_len), ecg_clean_sig, '-gD', markevery=r_peak, mfc='r')
plt.show()


# Step 4
# --------------------------------------------------------------------------------------
# Segment ecg record of 10 sec into  single heartbeats

cardiac_cycle_dict = nk.ecg_segment(ecg_clean_sig, rpeaks=r_peak, sampling_rate=100, show=True)
plt.show()

# Step 5
# --------------------------------------------------------------------------------------
# Plot each of the 12 cycles individually

keys = cardiac_cycle_dict.keys()
# dict_keys(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
no_of_cycles = len(keys)

cardiac_cycle_dict.values()
# Signal, Index and Label
# Values have a Dataframe structure with 3 columns

# cardiac_cycle_dict.items()  # pair of key and value

# Extract and plot each cycle

for i in range(1, no_of_cycles+1):
    cycle_i = cardiac_cycle_dict[str(i)]
    plt.plot(cycle_i['Index'], cycle_i['Signal'])
    print(i)
    plt.savefig("Graph" + str(i) + ".png", format="PNG")
    plt.show()



# Step 6 Identify other peaks P,Q,S,T peaks and Onset(P) and Offset(T)
# --------------------------------------------------------------------------------------
# Get only 1st R peak for 1st cycle

# r_peak_value = r_peak.flat[0]

ecg_opeaks_tuple = nk.ecg_delineate(ecg_clean_sig, rpeaks=r_peak, sampling_rate=100, method='peak', show=True,
                                    show_type='peaks', check=False)

# Tuple to dictionary
ecg_o_dict = ecg_opeaks_tuple[1]  # dictionary length 6 extract all key-value

ecg_p_peaks = ecg_o_dict['ECG_P_Peaks']
ecg_q_peaks = ecg_o_dict['ECG_Q_Peaks']
ecg_s_peaks = ecg_o_dict['ECG_S_Peaks']
ecg_r_peaks = list(r_peak)
ecg_t_peaks = ecg_o_dict['ECG_T_Peaks']

ecg_p_onset = ecg_o_dict['ECG_P_Onsets']
ecg_t_offset = ecg_o_dict['ECG_T_Offsets']


# Step 7
# --------------------------------------------------------------------------------------
# Plot P peaks

sig_len = len(ecg_clean_sig)

plt.plot(range(sig_len), ecg_clean_sig, '-gD', markevery=ecg_p_peaks, mfc='r')
plt.show()

# Plot P peaks in each cardiac cycle (cycles = 12, p-peaks = 12)
# Adding scatter plot for p peaks

for i in range(1, no_of_cycles+1):
    cycle_i = cardiac_cycle_dict[str(i)]
    plt.plot(cycle_i['Index'], cycle_i['Signal'])
    x = ecg_p_peaks[i-1]
    y = cycle_i.loc[cycle_i['Index'] == x, 'Signal'].iloc[0]
    plt.scatter(x, y, color='red', s=80, label='Peaks')
    plt.text(x, y, 'P', size = 20)
    x = ecg_t_peaks[i - 1]
    y = cycle_i.loc[cycle_i['Index'] == x, 'Signal'].iloc[0]
    plt.scatter(x, y, color='red', s=80)
    plt.text(x, y, 'T', size=20)
    x = ecg_q_peaks[i - 1]
    y = cycle_i.loc[cycle_i['Index'] == x, 'Signal'].iloc[0]
    plt.scatter(x, y, color='red', s=80)
    plt.text(x, y, 'Q', size=20)
    x = ecg_s_peaks[i - 1]
    y = cycle_i.loc[cycle_i['Index'] == x, 'Signal'].iloc[0]
    plt.scatter(x, y, color='red', s=80)
    plt.text(x, y, 'S', size=20)
    x = ecg_r_peaks[i - 1]
    y = cycle_i.loc[cycle_i['Index'] == x, 'Signal'].iloc[0]
    plt.scatter(x, y, color='red', s=80)
    plt.text(x, y, 'R', size=15)
    plt.legend(loc="upper right", prop={'size': 15})
    print(len(cycle_i))
    plt.show()




