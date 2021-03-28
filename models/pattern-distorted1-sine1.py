# Plot sine pattern # Replicate case of binary classification
# --------------------------------------------------------------#
# Plot just 1 sine wave of 1000 points and 10 sec duration so
# sampling time interval will be 10/1000 = .01 sec
# So sampling freq will be 1/.01 = 100 Hz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Step 1 - Draw 1 normal pattern
# --------------------------------------------------------------#

fi = 0.5  # Inherent freq 1 so 1 cycle time is 1/1 = 1 sec so 10 sec = 10 cycles
t = 10
fs = 100  # Sampling freq 100 so sampling time interval is 0.01
sample_points = 1000
ts = 0.01
a = 2

# 1000 Signal values as a function of 1000 time values
time = np.arange(0, 10, 0.01)

# Plot 1 sine pattern
sig1_sine = a * np.sin(2 * np.pi * fi * time)
# Create normal plots also with different a and f values?

# Introduce randomness to data
noise = np.random.normal(0, .1, sig1_sine.shape)
new_sig1_sine = sig1_sine + noise
print(new_sig1_sine)


# Plot this pattern
# Change ndarray to df
df = pd.DataFrame(new_sig1_sine, columns=['Signal'])
df.plot()
plt.show()


# Step 2 - Draw 1 abnormal random pattern
# --------------------------------------------------------------#
# 1% anomaly means 10/1000 points anomalous
# 3% anomaly means 30/1000 points anomalous
# 5% anomaly means 50/1000 points anomalous
# 8% anomaly means 80/1000 points anomalous
# 10% anomaly means 100/1000 points anomalous

# Make df_distort
# Choose starting location and see the distorted pattern
r = random.randint(0, 850)
print(r)

index_list = [range(r, r + 100, 1)]  # random starting point within pattern
df_distort = df.iloc[df.index[index_list]]  # random values in that index list

n = 50
# # Distortion method 1
# x = [random.randint(-1, 2) for _ in range(n)]
# print(x)
# df_distort1 = pd.DataFrame(x, columns=['Signal'])

# Distortion method 2

# Choose random number r between -3 to +3
r = random. randint(-3,3)
r1 = random. randint(-3,3)

x1 = [r] * n
x2 = [r1] * n
# Change list to ndarray and introduce randomness to data
x1_arr = np.array(x1)
x2_arr = np.array(x2)


noise1 = np.random.normal(0, .1, x1_arr.shape)
noise2 = np.random.normal(0, .1, x2_arr.shape)


new_x1 = x1_arr + noise1
print(new_x1)

new_x2 = x2_arr + noise2
print(new_x2)

# Convert arr to df
df_arr1 = pd.DataFrame(new_x1, columns=['Signal'])
df_arr2 = pd.DataFrame(new_x2, columns=['Signal'])

# Concatenate df_arr1 + df_arr2
frame = [df_arr1, df_arr2]
df_distort2 = pd.concat(frame)


# Copy df_distort to df_distort1
# Homogenize the index values
df_distort2.index = df_distort.index

# Delete the last row since when merging with normal last repeated
df_distort2 = df_distort2[:-1]

# Make dfn1 and dfn3
dfn1 = df.copy()
dfn3 = df.copy()

distort_f_index = df_distort.iloc[0]
print(distort_f_index)
distort_l_index = df_distort.iloc[-1]
print(distort_l_index)

# dfn1 to start from index 0 upto
dfn1 = dfn1.iloc[0:distort_f_index.name]
dfn3 = dfn3.iloc[distort_l_index.name:]

# Concatenate dfn1 + df_distort1 + dfn3
frames = [dfn1, df_distort2, dfn3]
new_df = pd.concat(frames)

# Plot new distorted df
new_df = new_df.astype(float)
new_df.plot()
plt.show()

