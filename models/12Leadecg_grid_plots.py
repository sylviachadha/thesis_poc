# Aim- Plot normal and IMI 12 lead ecg with grids
# ---------------------------------------------------

import wfdb

normal_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/00000/00001_lr'
record = wfdb.rdrecord(normal_file)
wfdb.plot_wfdb(record=record, title='Normal 1 ECG', ecg_grids='all', figsize=(22, 20))


imi_file = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/records100/00000/00008_lr'
record1 = wfdb.rdrecord(imi_file)
wfdb.plot_wfdb(record=record1, title='IMI 8 ECG', ecg_grids='all', figsize=(22, 20))

