# Plot 50 each sine normal and abnormal pattern
# --------------------------------------------------------------#
# Plot just 1 sine wave of 1000 points and 10 sec duration so
# sampling time interval will be 10/1000 = .01 sec
# So sampling freq will be 1/.01 = 100 Hz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

# Step 1 - Draw 50 normal pattern
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
sine_pattern = new_sig1_sine.copy()

# Crate 50 patterns

for n in range(49):
    new_row = a * np.sin(2 * np.pi * fi * time)
    noise1 = np.random.normal(0, .1, new_row.shape)
    new_pattern = new_row + noise1
    sine_pattern = np.vstack([sine_pattern, new_pattern])

# Plot this pattern (Change ndarray to df to plot)
# Change ndarray to df
df = pd.DataFrame(sine_pattern)

# Single Plot
df.iloc[0].plot()
plt.show()

# Put all images together to see patterns of normal synthetic sine wave
from PIL import Image
# All 50 normal plots
images_list = []
for x in range(49):
    f = plt.figure()
    df.iloc[x].plot()
    plt.show()
    # f.savefig('Sine Normal Pattern' + '.pdf', bbox_inches='tight')
    f.savefig('/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/' + str(x))
    image1 = Image.open(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/' + str(x) + '.png')
    im1 = image1.convert('RGB')
    images_list.append(im1)
    im1.save(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/normal_images.pdf',save_all=True, append_images=images_list)


# Step 2 - Draw 50 abnormal random pattern
# --------------------------------------------------------------#

df_a = df.copy()

tot_rows = len(df_a.index)
print(tot_rows)

anomalydf = pd.DataFrame()
for index, row in df_a.iterrows():
        print(row)
        r = random.randint(0, 850)
        print(r)
        index_list = [range(r, r + 100, 1)]
        print(index_list)
        distort_f_index = r
        print(distort_f_index)
        distort_l_index = r+100
        print(distort_l_index)
        n = 50
        r = random.randint(-3, 3)
        r1 = random.randint(-3, 3)

        x1 = [r] * n
        x2 = [r1] * n
        x1_arr = np.array(x1)
        x2_arr = np.array(x2)

        noise1 = np.random.normal(0, .1, x1_arr.shape)
        noise2 = np.random.normal(0, .1, x2_arr.shape)

        new_x1 = x1_arr + noise1
        new_x2 = x2_arr + noise2

        df_arr1 = pd.DataFrame(new_x1, columns=['Signal'])
        df_arr2 = pd.DataFrame(new_x2, columns=['Signal'])
        frame = [df_arr1, df_arr2]
        df_distort2 = pd.concat(frame)
        print(df_distort2)
        df_distort2.index = np.arange(start=distort_f_index, stop=distort_l_index, step=1)

        dfn1 = row.iloc[0:distort_f_index].to_frame(name="Signal")
        dfn3 = row.iloc[distort_l_index:].to_frame(name="Signal")

        # Concatenate dfn1 + df_distort1 + dfn3
        frames = [dfn1, df_distort2, dfn3]
        new_row = pd.concat(frames)
        new_row = new_row.transpose()
        anomalydf = anomalydf.append(new_row, ignore_index=True)


# Single Plot
anomalydf.iloc[10].plot()
plt.show()

# All Plots - Put all images together to see patterns of abnormal synthetic sine wave
from PIL import Image
# All 50 abnormal plots
imagesa_list = []
for x in range(tot_rows):
    f = plt.figure()
    anomalydf.iloc[x].plot()
    plt.show()
    # f.savefig('Sine Normal Pattern' + '.pdf', bbox_inches='tight')
    f.savefig('/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/' + str(x))
    image2 = Image.open(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/' + str(x) + '.png')
    im2 = image2.convert('RGB')
    imagesa_list.append(im2)
    im2.save(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/anomaly_images.pdf',save_all=True, append_images=imagesa_list)




