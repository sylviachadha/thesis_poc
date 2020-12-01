import wfdb

# wfdb introduction
# wfdb is a portable set of functions for reading and writing signal, annotation and
# header files in the formats used in PhysioBank.

# record = wfdb.rdrecord('data/108', sampto=21600)
# annotation = wfdb.rdann('data/108', 'atr', sampto=21600, summarize_labels=True)

# record and annotation objects created
record = wfdb.rdrecord('data/108')
annotation = wfdb.rdann('data/108', 'atr')

# wfdb plot
wfdb.plot_wfdb(record=record, annotation=annotation,
               title='Record 108 from MIT-BIH Arrhythmia Database',
               time_units='samples')

