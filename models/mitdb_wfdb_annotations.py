# Total 600K data annotations for each file [30 min]

# wfdb is a portable set of functions for reading and writing-
# 1. signal file,
# 2. annotation file,
# 3. header file
# used in the formats used in PhysioBank

# How to read record & annotation file? use wfdb.rdrecord and wfdb.rdann
# record = wfdb.rdrecord('data/108', sampto=21600)
# annotation = wfdb.rdann('data/108', 'atr', sampto=21600, summarize_labels=True)
# Here used annotation file

# Step 1 Import libraries ###########################
import wfdb
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"

# Step 2 Complete dataset (30 min data) ####################
annotation = wfdb.rdann('data/100', 'atr')
# Inside annotation object u have sample and symbol

df_annotated = pd.DataFrame() # empty dataframe
df_annotated['sample'] = annotation.sample[:]
df_annotated['label'] = annotation.symbol[:]
count = df_annotated['label'].value_counts()
print("Whole dataset annotations",count)

# Step 3 (5 min data) ############################
annotation = wfdb.rdann('data/100', 'atr')
df_lab = pd.DataFrame()
df_lab['sample'] = annotation.sample[:]
df_lab['label'] = annotation.symbol[:]

df_label = df_lab.iloc[:372]

count1 = df_label['label'].value_counts()
print("5 min data annotations",count1)


# Step 4 (1 min data) #############################

df_label = df_lab.iloc[:75]

count2 = df_label['label'].value_counts()
print("1 min data annotations",count2)










