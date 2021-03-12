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
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

Fs = 100 # Sampling freq
freq = 10 # Inherent freq
amp = 1
sample = 500  # For 50 cycles with 10 data points each
time = np.arange(0,5,0.01)

# For loop not required as time already a range function
signal_sine = amp * np.sin(2 * np.pi * freq * time)

plt.plot(time, signal_sine)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()

# Introduce randomness to data
noise = np.random.normal(0, .1, signal_sine.shape)
new_sine_signal = signal_sine + noise
print(new_sine_signal)

plt.plot(time, new_sine_signal)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()


# Step 2 Plot square or triangle wave
# -----------------------------------------------------------
# Generate Triangle wave with 50 patterns for Class 1
# with same time duration of 0.1 sec for each cycle so
# 5 sec for all 50 patterns
# -----------------------------------------------------------

#amp1 = 1   # change amplitude to 2
amp1 = 2
freq1 = 10

# When u give width=0.5 sawtooth becomes triangle wave
sig_triangle = amp1 * sg.sawtooth(2*np.pi*freq1*time, width=0.5)
plt.plot(time, sig_triangle)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

# Introduce randomness to data
noise = np.random.normal(0, .1, sig_triangle.shape)
new_triangle_signal = sig_triangle + noise
print(new_triangle_signal)

plt.plot(time, new_triangle_signal)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()


# Step 3 Plot together complete data
# -----------------------------------------------------------

# Original Merged signal
tot_time = np.arange(0,10,0.01) # Time duration to 10 sec for 50 cycles sine + 50 cycles square
tot_signal = np.concatenate((signal_sine, sig_triangle))

plt.plot(tot_time, tot_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude(V)')
plt.show()

# New merged signal
tot_new_signal = np.concatenate((new_sine_signal, new_triangle_signal))
plt.plot(tot_time, tot_new_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude(V)')
plt.show()


# Step 4 # Make dataset X_2D and y_1D # Features and labels
# -------------------------------------------------------------------

X_1D = tot_new_signal  # (Since values are going to form the features/X for the model)
X_2D = X_1D.reshape(100, 10)  # Change 1000 data points to 100 rows and 10 columns

# Change to df to see plot
df = pd.DataFrame(X_2D)

# Can see sine wave from index 0 to 49 and triangle from
# index 50 to 99
df1 = df.iloc[35]
plt.plot(df1)
plt.show()

# y is label
y_class0 = np.zeros(50)
y_class1 = np.ones(50)
y = np.concatenate((y_class0, y_class1))
y_1D = y

# print(X_2D)
# print(y_1D)

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








