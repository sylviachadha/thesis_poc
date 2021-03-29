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
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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

# # Put all images together to see patterns of normal synthetic sine wave
# from PIL import Image
# # All 50 normal plots
# images_list = []
# for x in range(49):
#     f = plt.figure()
#     df.iloc[x].plot()
#     plt.show()
#     # f.savefig('Sine Normal Pattern' + '.pdf', bbox_inches='tight')
#     f.savefig('/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/' + str(x))
#     image1 = Image.open(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/' + str(x) + '.png')
#     im1 = image1.convert('RGB')
#     images_list.append(im1)
#     im1.save(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/normal_images/normal_images.pdf',save_all=True, append_images=images_list)
#

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

# # All Plots - Put all images together to see patterns of abnormal synthetic sine wave
# from PIL import Image
# # All 50 abnormal plots
# imagesa_list = []
# for x in range(tot_rows):
#     f = plt.figure()
#     anomalydf.iloc[x].plot()
#     plt.show()
#     # f.savefig('Sine Normal Pattern' + '.pdf', bbox_inches='tight')
#     f.savefig('/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/' + str(x))
#     image2 = Image.open(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/' + str(x) + '.png')
#     im2 = image2.convert('RGB')
#     imagesa_list.append(im2)
#     im2.save(r'/Users/sylviachadha/Desktop/Tools/PyCharm/thesis_poc/anomaly_images/anomaly_images.pdf',save_all=True, append_images=imagesa_list)


# Apply ML Algorithms for Supervised learning
# df has 50 normal patterns
# anomalydf has 50 abnormal patterns
# Most ML Algorithms accept ndarray instead of dataframe


# AIM - To run ML Supervised algorithms on Sine
# normal and abnormal data

# ------------------------------
# STEP 1 Change df to ndarray
# ------------------------------
df_arr = df.to_numpy()
anomalydf_arr = anomalydf.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D = np.concatenate((df_arr, anomalydf_arr))


# -------------------------------------------
# STEP 2 See plots to verify 2D Input data
# -------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and triangle from
# index 50 to 99
X_2D_df1 = X_2D_df.iloc[52]
plt.plot(X_2D_df1)
plt.show()

# -------------------------------------------
# STEP 3 Data to input to ML Algorithms
# X_2D and y_1D
# --------------------------------------------

# X_2D as created above
# y is label
y_class0 = np.zeros(50)
y_class1 = np.ones(50)
y = np.concatenate((y_class0, y_class1))
y_1D = y

# -------------------------------------------------------------------
# Step 4  Reduce dataset X_2D(53,1000) and y_1D(53) # Features and labels
# All Class 0 and only 3 samples of Class 1
# Use this 94.3% Class 0 and 5.7% Class 1 for Training
# -------------------------------------------------------------------

# Reduce Abnormal class to just 3 samples by removing 47 observations
# from tail of X and Y which are currently 100

# ** Need to try different combinations of 3 samples which are
# ** given for training of anomaly class

n = 47
X_short = X_2D[:-n, :]  # :-n will include all rows starting from 0 upto excluding the last 47 rows
print("Filtered data", X_short)
print("Length of Filtered Data", len(X_short))

# Remove 47 observations from tail of Y
y_short = np.delete(y_1D, range(53, 100, 1))

# -------------------------------------------------------------------
# STEP 5 - Split into Train & Test to feed to ML Algorithm
# ----------------------------------------------------------
# Check Shape of data and target/label
print("Shape of data", X_short.shape)
print("Shape of label", y_short.shape)

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_short, y_short, test_size=0.2) #random_state=4)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

# -----------------------------------------------------------------#
# STEP 6 - One CLass methods / Unsupervised methods preprocessing
# -----------------------------------------------------------------#
# Require only normal data for training but require both normal &
# anomalous data for test

X_train_1class = X_train[y_train == 0]

# -----------------------------------------------------------------#
# STEP 7 - Modelling #1. Import 2.Instantiate  3. Fit  4. Predict
# -----------------------------------------------------------------#
# Input ready for model as in previous step-
# X_train_1class

# Model 1 # One Class SVM
# --------------------------------------------------------------

# A. Import
from sklearn.svm import OneClassSVM

# B. Instantiate
ocsvm = OneClassSVM(gamma='scale', nu=0.05)

# C. Train
ocsvm.fit(X_train_1class)

# D. Predict
ypred1 = ocsvm.predict(X_test)
#cprint("y_pred", ypred1)

# Model 2 # Isolation Forest
# --------------------------------------------------------------

# A. Import
from sklearn.ensemble import IsolationForest

# B. Instantiate
iforest = IsolationForest(contamination=0.05)

# C. Train on Majority class only
iforest.fit(X_train_1class)

# D. Predict
ypred2 = iforest.predict(X_test)
# print("y_pred", ypred2)


# -----------------------------------------------------------------------------
# Step 6 - O/P of 1 class - Preprocessing before evaluation
# -----------------------------------------------------------------------------
# Change actual y_test to 1 & -1 as y_pred is returned in this form
# Mark inliers 1, outliers -1 in test_y


y_test[y_test == 1] = -1
y_test[y_test == 0] = 1
# print("y_test", y_test_short_num)

# -----------------------------------------
# Step 7 - Evaluation Metrics
# -----------------------------------------

# Define generalized function

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return(cm, acc, recall, precision, f1score)

# Model 1 Logistic Regression
ocsvm_result = evaluate_model(y_test, ypred1)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result)
ocsvm.report = classification_report(y_test, ypred1)
print(ocsvm.report)

# Model 2 Isolation Forest
iforest_result = evaluate_model(y_test, ypred2)
print("Isolation Forest values for cm, accuracy, recall, precision and f1 score", iforest_result)
iforest.report = classification_report(y_test, ypred2)
print(iforest.report)


# ---------------------------------------------------------------------
# PART B - TOTAL 53 DATA = 42 TRAIN DATA AND 11 TEST DATA
# ---------------------------------------------------------------------
# 20 TEST DATA added from minority class from remaining 47 samples
# which model has not seen
# ---------------------------------------------------------------------
# STEP 1 - Make another test set with more anomaly
# --------------------------------------------------------------------

# Step 1 Make y
y_1D_new1 = y_1D.copy() # original y which has 100 labels
y_1D_new1[y_1D_new1 == 1] = -1
y_1D_new1[y_1D_new1 == 0] = 1

y_test_xl1 = y_test.copy()

y_test_xl2 = np.delete(y_1D_new1, range(0, 80, 1))

y_test_merged1 = np.concatenate((y_test_xl1, y_test_xl2))
print("Merged labels of new test set", y_test_merged1)

# Step 2 Make X
X_2D_new1 = X_2D.copy()
X_test_xl1 = X_test.copy()

# Keep 40 observations from tail of X
n = 20
X_2D_new1 = X_2D_new1[-n:, :]
print("Filtered data", X_2D_new1)
print("Length of Filtered Data", len(X_2D_new1))

# Merge X_test_xl and new_data_X1_copy
X_test_merged1 = np.concatenate((X_test_xl1, X_2D_new1))
print("Length of new test set", len(X_test_merged1))

# STEP 2 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset

# Model 1
ypred3 = ocsvm.predict(X_test_merged1)  # Predict
ocsvm_result1 = evaluate_model(y_test_merged1, ypred3)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result1)

# Model 2 Isolation Forest
ypred4 = iforest.predict(X_test_merged1)  # Predict
iforest_result1 = evaluate_model(y_test_merged1, ypred4)
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result1)




# ---------------------------------------------------------------------
# PART C - TOTAL 53 DATA = 42 TRAIN DATA AND 11 TEST DATA
# ---------------------------------------------------------------------
# 40 TEST DATA added from minority class from remaining 47 samples
# which model has not seen
# ---------------------------------------------------------------------
# STEP 1 - Make another test set with more anomaly
# --------------------------------------------------------------------

# Step 1 Make y
y_1D_new1 = y_1D.copy() # original y which has 100 labels
y_1D_new1[y_1D_new1 == 1] = -1
y_1D_new1[y_1D_new1 == 0] = 1

y_test_xl1 = y_test.copy()

y_test_xl2 = np.delete(y_1D_new1, range(0, 60, 1))

y_test_merged1 = np.concatenate((y_test_xl1, y_test_xl2))
print("Merged labels of new test set", y_test_merged1)

# Step 2 Make X
X_2D_new1 = X_2D.copy()
X_test_xl1 = X_test.copy()

# Keep 40 observations from tail of X
n = 40
X_2D_new1 = X_2D_new1[-n:, :]
print("Filtered data", X_2D_new1)
print("Length of Filtered Data", len(X_2D_new1))

# Merge X_test_xl and new_data_X1_copy
X_test_merged1 = np.concatenate((X_test_xl1, X_2D_new1))
print("Length of new test set", len(X_test_merged1))

# STEP 2 - Predictions and Evaluation results on enhanced test set
# -------------------------------------------------------------------
# Take predictions on augmented test dataset

# Model 1
ypred3 = ocsvm.predict(X_test_merged1)  # Predict
ocsvm_result1 = evaluate_model(y_test_merged1, ypred3)
print("One class svm values for cm, accuracy, recall, precision and f1 score", ocsvm_result1)

# Model 2 Isolation Forest
ypred4 = iforest.predict(X_test_merged1)  # Predict
iforest_result1 = evaluate_model(y_test_merged1, ypred4)
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result1)

