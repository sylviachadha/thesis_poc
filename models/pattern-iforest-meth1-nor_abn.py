# PART A 50 normal and 50 abnormal - (Train/ Test both)
# --------------------------------------------------------------
# Aim - iforest method of anomaly detection on simulated data
# --------------------------------------------------------------#
# Reference https://heartbeat.fritz.ai/isolation-forest-algorithm-for-anomaly-detection-2a4abd347a5

# BALANCED (Model fit on balanced classes, predict on different dataset)
# Step 1 - Import libraries
# ----------------------------#
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


# Step 2A - Draw 50 normal pattern
# --------------------------------------------------------------#

def sine_normal_function():
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

    return sine_pattern


# Plot this pattern (Change ndarray to df to plot)
# Change ndarray to df

# Call function

sine_pattern = sine_normal_function()

# Change to df
df = pd.DataFrame(sine_pattern)

# Single Plot
df.iloc[0].plot()
plt.show()


# Step 2B - Draw 50 abnormal random pattern
# --------------------------------------------------------------#

def sine_abnormal_function():
    df_a = df.copy()

    tot_rows = len(df_a.index)
    print(tot_rows)

    anomaly_df = pd.DataFrame()
    for index, row in df_a.iterrows():
        print(row)
        r = random.randint(0, 850)
        print(r)
        index_list = [range(r, r + 100, 1)]
        print(index_list)
        distort_f_index = r
        print(distort_f_index)
        distort_l_index = r + 100
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
        anomaly_df = anomaly_df.append(new_row, ignore_index=True)
    return anomaly_df


# Call function
anomaly_df = sine_abnormal_function()

# Single Plot
anomaly_df.iloc[10].plot()
plt.show()

# STEP 3 - Create Data to input to Algorithm X_2D and y_1D
# --------------------------------------------------------
# Create x_2D
# Change df and
df_arr = df.to_numpy()
anomaly_df_arr = anomaly_df.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D = np.concatenate((df_arr, anomaly_df_arr))

# Create y_1D (labels)
y_class0 = np.zeros(50)
y_class1 = np.ones(50)
y = np.concatenate((y_class0, y_class1))
y_1D = y
y_1D_df = pd.DataFrame(y_1D, columns=['actual'])

# STEP 4 - Qualitative Check to verify 2D Input data
# -------------------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 99
X_2D_plot = X_2D_df.iloc[52]
plt.plot(X_2D_plot)
plt.show()

# STEP 5 - Define, fit and predict
# ---------------------------------------------------
# Define a model variable and instantiate the isolation forest class.

# Main Parameters
# Contamination is most important because predict method makes use of the threshold,
# which is set by the contamination value.
max_features = 1
n_estimators = 100
max_samples = 'auto'
contamination = float(0.50)

# Instantiate / Define
iforest_model = IsolationForest(max_features=max_features,
                                n_estimators=n_estimators,
                                max_samples=max_samples,
                                contamination=contamination)

# Fit
iforest_model.fit(X_2D_df)

# X_2D_df_res = X_2D_df.copy()
#
# X_2D_df_res['actual'] = y_1D_df
# X_2D_df_res['actual'].replace({0: 1, 1: -1}, inplace=True)
#
# # Bypassing the X_2D as a parameter to decision_function() we can find the
# # values of the scores column.
# X_2D_df_res['scores'] = iforest_model.decision_function(X_2D_df)

# We can find the values of the anomaly column bypassing the X_2D as a parameter
# to predict() the function of the trained model.
# X_2D_df_res['pred'] = iforest_model.predict(X_2D_df)


# Predict
# Make test data to predict

sine_pattern_test = sine_normal_function()
# Change to df
df_test = pd.DataFrame(sine_pattern_test)

# Single Plot
df_test.iloc[0].plot()
plt.show()


# Call function abnormal
anomaly_df_test = sine_abnormal_function()

# Single Plot
anomaly_df_test.iloc[10].plot()
plt.show()

# Concatenate test Data to input to Algorithm X_2D_test
# --------------------------------------------------------
# Create x_2D
# Change df and
df_arr_test = df_test.to_numpy()
anomaly_df_arr_test = anomaly_df_test.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D_test = np.concatenate((df_arr_test, anomaly_df_arr_test))

#y_1D_df_test = pd.DataFrame(y_1D, columns=['actual'])

# Check to verify 2D Input data
# -------------------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df_test = pd.DataFrame(X_2D_test)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 99
X_2D_test_plot = X_2D_df_test.iloc[52]
plt.plot(X_2D_test_plot)
plt.show()

X_2D_df_test_res = X_2D_df_test.copy()
X_2D_df_test_res['actual'] = y_1D_df
X_2D_df_test_res['actual'].replace({0: 1, 1: -1}, inplace=True)


# Add scores and make predictions
X_2D_df_test_res['scores'] = iforest_model.decision_function(X_2D_df_test)
X_2D_df_test_res['pred'] = iforest_model.predict(X_2D_df_test)



# STEP 6 - Evaluate model
# ---------------------------------------------------
def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred, pos_label=-1)
    precision = precision_score(y_actual, y_pred, pos_label=-1)
    f1score = f1_score(y_actual, y_pred, pos_label=-1)
    return (cm, acc, recall, precision, f1score)


iforest_result = evaluate_model(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result)
iforest_model.report = classification_report(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print(iforest_model.report)

# -----------------------------------------------------------------------------------------------#

# PART B 50 normal and 3 abnormal - (Train/ Test both)
# --------------------------------------------------------------
# IMBALANCED (Model fit on imbalanced classes, predict on different dataset)
# 50 NORMAL AND 3 ABNORMAL
# -----------------------------------------------------------------------------------------------#

# Aim - iforest method of anomaly detection on simulated data
# --------------------------------------------------------------#
# Reference https://heartbeat.fritz.ai/isolation-forest-algorithm-for-anomaly-detection-2a4abd347a5

# Step 1 - Import libraries
# ----------------------------#
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

# Step 2A - Draw 50 normal pattern
# --------------------------------------------------------------#

def sine_nor_func():
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

    return sine_pattern

# Call function
sine_pattern = sine_nor_func()

# Plot this pattern (Change ndarray to df to plot)
# Change ndarray to df
df = pd.DataFrame(sine_pattern)

# Single Plot
df.iloc[0].plot()
plt.show()

# Step 2B - Draw 3 abnormal random pattern
# --------------------------------------------------------------#

def sine_abnorm():
    df_a = df.head(3)

    tot_rows = len(df_a.index)
    print(tot_rows)

    anomaly_df = pd.DataFrame()
    for index, row in df_a.iterrows():
        print(row)
        r = random.randint(0, 850)
        print(r)
        index_list = [range(r, r + 100, 1)]
        print(index_list)
        distort_f_index = r
        print(distort_f_index)
        distort_l_index = r + 100
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
        anomaly_df = anomaly_df.append(new_row, ignore_index=True)

    return anomaly_df

# Call function
anomaly_df = sine_abnorm()

# Single Plot
anomaly_df.iloc[2].plot()
plt.show()

# STEP 3 - Create Data to input to Algorithm X_2D # Labels not required by algo
# -------------------------------------------------------------------------------
# Create x_2D
# Change df and
df_arr = df.to_numpy()
anomaly_df_arr = anomaly_df.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D = np.concatenate((df_arr, anomaly_df_arr))

# Create y_1D (labels)
y_class0 = np.zeros(50)
y_class1 = np.ones(3)
y = np.concatenate((y_class0, y_class1))
y_1D = y
y_1D_df = pd.DataFrame(y_1D, columns=['actual'])


# STEP 4 - Qualitative Check to verify 2D Input data
# -------------------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 99
X_2D_plot = X_2D_df.iloc[52]
plt.plot(X_2D_plot)
plt.show()

# # STEP 5 - Create imbalance in data
# # -----------------------------------------
#
# # Remove last 47 rows of a dataframe
# n = 47
# X_2D = X_2D[:-n, :]
# X_2D_df1 = pd.DataFrame(X_2D)
#
# # Remove last 47 rows of a dataframe
# y_1D = np.delete(y_1D, range(53, 100, 1))
# y_1D_df = pd.DataFrame(y_1D, columns=['actual'])

# STEP 6 - Define, fit and predict
# ---------------------------------------------------
# Define a model variable and instantiate the isolation forest class.

# Main Parameters
# Contamination is most important because predict method makes use of the threshold,
# which is set by the contamination value.
max_features = 1
n_estimators = 100
max_samples = 'auto'
contamination = float(0.05)

# Instantiate / Define
iforest_model = IsolationForest(max_features=max_features,
                                n_estimators=n_estimators,
                                max_samples=max_samples,
                                contamination=contamination)

# Fit
iforest_model.fit(X_2D_df)

# #Predict
# X_2D_df_res = X_2D_df.copy()
#
# X_2D_df_res['actual'] = y_1D_df
# X_2D_df_res['actual'].replace({0: 1, 1: -1}, inplace=True)
#
# # Bypassing the X_2D as a parameter to decision_function() we can find the
# # values of the scores column.
# X_2D_df_res['scores'] = iforest_model.decision_function(X_2D_df)
#
# # We can find the values of the anomaly column bypassing the X_2D as a parameter
# # to predict() the function of the trained model.
# X_2D_df_res['pred'] = iforest_model.predict(X_2D_df)



# Predict (Get predictions on new dataset)
# Call sine normal & abnormal functions and concatenate them

sine_pattern_test = sine_nor_func()
df_test = pd.DataFrame(sine_pattern_test)

anomaly_df_test = sine_abnorm()

# Concatenate and change to df and see plot
df_arr_test = df_test.to_numpy()
anomaly_df_test_arr = anomaly_df_test.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D_test = np.concatenate((df_arr_test, anomaly_df_test_arr))
X_2D_df_test = pd.DataFrame(X_2D_test)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 52
X_2D_plot_test = X_2D_df_test.iloc[51]
plt.plot(X_2D_plot_test)
plt.show()

# actual, scores and predict on test data
X_2D_df_test_res = X_2D_df_test.copy()
#
X_2D_df_test_res['actual'] = y_1D_df
X_2D_df_test_res['actual'].replace({0: 1, 1: -1}, inplace=True)

X_2D_df_test_res['scores'] = iforest_model.decision_function(X_2D_df_test)
X_2D_df_test_res['pred'] = iforest_model.predict(X_2D_df_test)


# STEP 7 - Evaluate model
# ---------------------------------------------------
def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred, pos_label=-1)
    precision = precision_score(y_actual, y_pred, pos_label=-1)
    f1score = f1_score(y_actual, y_pred, pos_label=-1)
    return (cm, acc, recall, precision, f1score)


iforest_result = evaluate_model(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result)
iforest_model.report = classification_report(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print(iforest_model.report)

# -----------------------------------------------------------------------------------------------#
# PART C 50 normal and 15 abnormal - (Train/ Test both)
# -----------------------------------------------------------------------------------------------#

# IMBALANCED (Model fit on imbalanced classes, predict on different dataset)
# 50 NORMAL AND 15 ABNORMAL

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

# Step 2A - Draw 50 normal pattern
# --------------------------------------------------------------#

def sine_nor_func():
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

    return sine_pattern

# Call function
sine_pattern = sine_nor_func()

# Plot this pattern (Change ndarray to df to plot)
# Change ndarray to df
df = pd.DataFrame(sine_pattern)

# Single Plot
df.iloc[0].plot()
plt.show()

# Step 2B - Draw 3 abnormal random pattern
# --------------------------------------------------------------#

def sine_abnorm():
    df_a = df.head(15)

    tot_rows = len(df_a.index)
    print(tot_rows)

    anomaly_df = pd.DataFrame()
    for index, row in df_a.iterrows():
        print(row)
        r = random.randint(0, 850)
        print(r)
        index_list = [range(r, r + 100, 1)]
        print(index_list)
        distort_f_index = r
        print(distort_f_index)
        distort_l_index = r + 100
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
        anomaly_df = anomaly_df.append(new_row, ignore_index=True)

    return anomaly_df

# Call function
anomaly_df = sine_abnorm()

# Single Plot
anomaly_df.iloc[2].plot()
plt.show()

# STEP 3 - Create Data to input to Algorithm X_2D # Labels not required by algo
# -------------------------------------------------------------------------------
# Create x_2D
# Change df and
df_arr = df.to_numpy()
anomaly_df_arr = anomaly_df.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D = np.concatenate((df_arr, anomaly_df_arr))

# Create y_1D (labels)
y_class0 = np.zeros(50)
y_class1 = np.ones(15)
y = np.concatenate((y_class0, y_class1))
y_1D = y
y_1D_df = pd.DataFrame(y_1D, columns=['actual'])


# STEP 4 - Qualitative Check to verify 2D Input data
# -------------------------------------------------------
# Change to df to see plot to verify normal & abnormal
X_2D_df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 99
X_2D_plot = X_2D_df.iloc[64]
plt.plot(X_2D_plot)
plt.show()

# # STEP 5 - Create imbalance in data
# # -----------------------------------------
#
# # Remove last 47 rows of a dataframe
# n = 47
# X_2D = X_2D[:-n, :]
# X_2D_df1 = pd.DataFrame(X_2D)
#
# # Remove last 47 rows of a dataframe
# y_1D = np.delete(y_1D, range(53, 100, 1))
# y_1D_df = pd.DataFrame(y_1D, columns=['actual'])

# STEP 6 - Define, fit and predict
# ---------------------------------------------------
# Define a model variable and instantiate the isolation forest class.

# Main Parameters
# Contamination is most important because predict method makes use of the threshold,
# which is set by the contamination value.
max_features = 1
n_estimators = 100
max_samples = 'auto'
contamination = float(0.20)

# Instantiate / Define
iforest_model = IsolationForest(max_features=max_features,
                                n_estimators=n_estimators,
                                max_samples=max_samples,
                                contamination=contamination)

# Fit
iforest_model.fit(X_2D_df)

# #Predict
# X_2D_df_res = X_2D_df.copy()
#
# X_2D_df_res['actual'] = y_1D_df
# X_2D_df_res['actual'].replace({0: 1, 1: -1}, inplace=True)
#
# # Bypassing the X_2D as a parameter to decision_function() we can find the
# # values of the scores column.
# X_2D_df_res['scores'] = iforest_model.decision_function(X_2D_df)
#
# # We can find the values of the anomaly column bypassing the X_2D as a parameter
# # to predict() the function of the trained model.
# X_2D_df_res['pred'] = iforest_model.predict(X_2D_df)



# Predict (Get predictions on new dataset)
# Call sine normal & abnormal functions and concatenate them

sine_pattern_test = sine_nor_func()
df_test = pd.DataFrame(sine_pattern_test)

anomaly_df_test = sine_abnorm()

# Concatenate and change to df and see plot
df_arr_test = df_test.to_numpy()
anomaly_df_test_arr = anomaly_df_test.to_numpy()

# Concatenate total data (normal + abnormal)
X_2D_test = np.concatenate((df_arr_test, anomaly_df_test_arr))
X_2D_df_test = pd.DataFrame(X_2D_test)

# Can see sine wave from index 0 to 49 and abnormal wave from
# index 50 to 52
X_2D_plot_test = X_2D_df_test.iloc[51]
plt.plot(X_2D_plot_test)
plt.show()

# actual, scores and predict on test data
X_2D_df_test_res = X_2D_df_test.copy()
#
X_2D_df_test_res['actual'] = y_1D_df
X_2D_df_test_res['actual'].replace({0: 1, 1: -1}, inplace=True)

X_2D_df_test_res['scores'] = iforest_model.decision_function(X_2D_df_test)
X_2D_df_test_res['pred'] = iforest_model.predict(X_2D_df_test)


# STEP 7 - Evaluate model
# ---------------------------------------------------
def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred, pos_label=-1)
    precision = precision_score(y_actual, y_pred, pos_label=-1)
    f1score = f1_score(y_actual, y_pred, pos_label=-1)
    return (cm, acc, recall, precision, f1score)


iforest_result = evaluate_model(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print("iForest values for cm, accuracy, recall, precision and f1 score", iforest_result)
iforest_model.report = classification_report(X_2D_df_test_res['actual'], X_2D_df_test_res['pred'])
print(iforest_model.report)







