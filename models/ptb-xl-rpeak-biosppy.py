# Install Packages
from biosppy.signals import ecg
import wfdb
import matplotlib.pyplot as plt


# Random Check for Normal (109,21831, 5298) for all 12 leads
normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/21000/21831_lr'
record = wfdb.rdrecord(normal_file, channels=[2])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()

cardiac_cycles, rpeaks = ecg.extract_heartbeats(signal=record.p_signal.flatten(),
                                                rpeaks=r_peaks,
                                                sampling_rate=record.fs,
                                                before=0.2,
                                                after=0.4)


# Random Check for AbNormal IMI Patient (2134,11285,1577] for all 12 leads
imi_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/01000/01577_lr'
record = wfdb.rdrecord(imi_file, channels=[11])
out = ecg.ecg(signal=record.p_signal.flatten(), sampling_rate=record.fs, show=False)

r_peaks = out[2]
heart_rate = out[6]

plt.plot(range(record.sig_len), record.p_signal, '-gD', markevery=r_peaks, mfc='r')
plt.show()