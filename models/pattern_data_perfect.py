# -----------------------------------------------------------
# AIM - Generate Synthetic pattern data to test
# existing supervised and one class/unsupervised methods
# tested on iris dataset
# -----------------------------------------------------------

# Step 1
# -----------------------------------------------------------

# Generate Sine Wave with 50 patterns, so we have
# 50 observations for sine class i.e Class 0
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy import signal

Fs = 100
# Sampling freq is 100 means sampling interval time is 0.01
freq = 10
# Inherent freq is 10 so completes 1 full cycle in 0.1 sec

# Completes 1 cycle in 0.1 sec and sampling interval is 0.01
# 0.1 / 0.01 so 10 data points

amp = 1
sample = 500  # For 50 cycles with 10 data points each
time = np.arange(0,5,0.01)

# For loop not required as time already a range function
signal_sine = amp * np.sin(2 * np.pi * freq * time)

plt.plot(time, signal_sine)
# plt.plot(x, y, 'r')  # if need plot for a specific colour
plt.xlabel('time')
plt.ylabel('signal')
plt.show()


# Step 2 Plot square or triangle wave
# -----------------------------------------------------------
# Generate Square wave with 50 patterns for Class 1
# with same time duration of 0.1 sec for each cycle so
# 5 sec for all 50 patterns
# -----------------------------------------------------------
#
# # Case 1
# amp1 = 1
# freq1 = 10
#
# sig_square = amp1 * sg.square(2 * np.pi * freq1 * time)
#
# plt.plot(time, sig_square)
# plt.xlabel('Time')
# plt.ylabel('Amplitude(V)')
# plt.show()


# # Case 2
# amp1 = 2
# freq1 = 5
# value1 = []
# # So this will cover 1 cycle in 1/5 = 0.2 seconds - 25 cycles
# # To have 50 cycles for this we need to increase time
# # duration to 10 seconds instead of 5 sec.

# sig_square = amp1*sg.square(2*np.pi*freq1*time)

# plt.plot(x, sig_square)
# plt.xlabel('Time')
# plt.ylabel('Amplitude(V)')
# plt.show()


# Plot triangle wave
amp1 = 1
freq1 = 10

sig_sawtooth = amp1 * sg.sawtooth(2*np.pi*freq1*time, width=0.5)
plt.plot(time, sig_sawtooth)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()


# Step 3 Plot together complete data
# -----------------------------------------------------------

tot_time = np.arange(0,10,0.01) # Time duration to 10 sec for 50 cycles sine + 50 cycles square

#tot_signal = np.concatenate((signal_sine, sig_square))
tot_signal = np.concatenate((signal_sine, sig_sawtooth))


plt.plot(tot_time, tot_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude(V)')
plt.show()


# Step 4 # Make dataset X_2D and y_1D # Features and labels
# -------------------------------------------------------------------
# y_complete is ndarray of 1000 points which need to split into rows
# of 10 points each, so total should be 100 rows (100,10)

X_1D = tot_signal  # (Since values are going to form the features/X for the model)
X_2D = X_1D.reshape(100, 10)  # Change 1000 data points to 100 rows and 10 columns

# Change to df to see plot
df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and square from
# index 50 to 99
df1 = df.iloc[41]
plt.plot(df1)
plt.show()

# y is label
y_class0 = np.zeros(50)
y_class1 = np.ones(50)
y = np.concatenate((y_class0, y_class1))
y_1D = y

print(X_2D)
print(y_1D)

# Step 5 Train and Test Split
# -----------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2D, y_1D, test_size=0.2, random_state=4,stratify=y_1D)

# print shapes of new X objects
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

# print shapes of new y objects
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

# Step 6 Modelling # 1. Import 2. Instantiate 3. Fit 4. Predict
# --------------------------------------------------------------

# # Model 1 - Logistic Regression
# #------------------------------------------------------------------
#
from sklearn.linear_model import LogisticRegression # Import
logreg = LogisticRegression(max_iter=200) # Instantiate
logreg.fit(X_2D,y_1D) # Fit
y_pred_lr = logreg.predict(X_test) # Predict

# Model 2 - KNN
#------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier # Import
knn = KNeighborsClassifier(n_neighbors=5) # Instantiate
knn.fit(X_train,y_train) # Fit
y_pred_knn = knn.predict(X_test) # Predict

# Model 3 - Decision Tree
#------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier # Import
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # Instantiate
dt.fit(X_train, y_train) # Fit
y_pred_dt = dt.predict(X_test) # Predict

# Model 4 - Random Forest
#------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier # Import
rf = RandomForestClassifier(n_estimators=200) # Instantiate
rf.fit(X_train, y_train) # Fit
y_pred_rf = rf.predict(X_test) # Predict

# Model 5 - SVM - Linear
#------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc = SVC()  # Instantiate
svc.fit(X_train, y_train)  # Fit
y_pred_svc = svc.predict(X_test)  # Predict

# Model 6 - SVM - Non-Linear
#------------------------------------------------------------------

from sklearn.svm import SVC  # Import
svc_rbf = SVC(kernel='rbf', gamma=1.0)  # Instantiate
svc_rbf.fit(X_train, y_train)  # Fit
y_pred_svcrbf = svc_rbf.predict(X_test)  # Predict

# Model 7 - Naive Bayes
#------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB  # Import
nbayes = GaussianNB()  # Instantiate
nbayes.fit(X_train, y_train)  # Fit (features, labels)
y_pred_nb = nbayes.predict(X_test)  # Predict


# STEP 7 - Evaluation Metrics
#------------------------------------------------------------------

def evaluate_model(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    print(cm)
    acc = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    f1score = f1_score(y_actual, y_pred)
    return(cm, acc, recall, precision, f1score)

# Model 1 Logistic Regression

lr_result = evaluate_model(y_test, y_pred_lr)
print("Logistic Regression values for cm, accuracy, recall, precision and f1 score", lr_result)

# Model 2 - KNN

knn_result = evaluate_model(y_test, y_pred_knn)
print("KNN values for cm, accuracy, recall, precision and f1 score", knn_result)

# Model 3 - Decision Tree

dt_result = evaluate_model(y_test, y_pred_dt)
print("Decision Tree values for cm, accuracy, recall, precision and f1 score", dt_result)

# Model 4 - Random Forest

rf_result = evaluate_model(y_test, y_pred_rf)
print("Random Forest values for cm, accuracy, recall, precision and f1 score", rf_result)

# Model 5 - SVM - Linear

svclinear_result = evaluate_model(y_test, y_pred_svc)
print("SVM Linear values for cm, accuracy, recall, precision and f1 score", svclinear_result)

# Model 6 - SVM - Non-Linear

svcrbf_result = evaluate_model(y_test, y_pred_svcrbf)
print("SVM RBF Kernel values for cm, accuracy, recall, precision and f1 score", svcrbf_result)

# Model 7 - Naive Bayes

nb_result = evaluate_model(y_test, y_pred_nb)
print("Naive Bayes values for cm, accuracy, recall, precision and f1 score", nb_result)








