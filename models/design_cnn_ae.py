# https://keras.io/examples/timeseries/timeseries_anomaly_detection/
# Step 1 Import Libraries
# ----------------------------------------------------------------
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt



# Step 2 Load Data
# ----------------------------------------------------------------
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_db_normal.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Select only combination of NORM and SR as NORM

def aggregate_diagnostic1(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)

print(Y['diagnostic_superclass'])

print(len(Y))

# Select only 1 lead data which is aVF lead/Channel 5
# aVF lead
X_aVF = X[:, :, 5]

print(X_aVF.shape)

# Since AE expects 3D input (batch_size, sequence_length, num_features)
# Reshape from 2 dimensional to 3 dimensional

# batch_size = X_aVF.shape[0] which is 4699
# sequence_length = X_aVF.shape[1] which is 1000
# num_features =  1 which is just lead avF

from numpy import newaxis
X_aVF_3D = X_aVF[:, :, newaxis]
print(X_aVF_3D.shape)
print(X_aVF_3D.shape[0])  # 4699
print(X_aVF_3D.shape[1])  # 1000
print(X_aVF_3D.shape[2])  # 1


# Step 3 Reconstruction Convolutional Autoencoder model to detect
# anomalies in Timeseries data
# ----------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Input(shape=(X_aVF_3D.shape[1], X_aVF_3D.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=10, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=10, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=10, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=10, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=10, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# Step 4
# ------------------------------------------------------------------------------------------
# We are using x_train as both the input and the target since this is a reconstruction model.

history = model.fit(
    X_aVF_3D,
    X_aVF_3D,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

# Step 5 Learning curve evolution to detect model behaviour
# -------------------------------------------------------------------
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# Step 6 Get train MAE loss
# -------------------------------------------------------------------
x_train_pred = model.predict(X_aVF_3D)
train_mae_loss = np.mean(np.abs(x_train_pred - X_aVF_3D), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)


# Step 7 Compare Reconstruction
# ------------------------------------------
# Checking how the first sequence is learnt
plt.plot(X_aVF_3D[4])
plt.plot(x_train_pred[4])
plt.show()


# Step 8 Prepare Test Data
# ------------------------------------------

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_imi.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)

print(Y['diagnostic_superclass'])

print(len(Y))
X_aVF_test = X[:, :, 5]
print(X_aVF_test.shape)

from numpy import newaxis
X_aVF_test_3D = X_aVF_test[:, :, newaxis]
print(X_aVF_test_3D.shape)
print(X_aVF_test_3D.shape[0])  # 25
print(X_aVF_test_3D.shape[1])  # 1000
print(X_aVF_test_3D.shape[2])  # 1


# Step 9 Get test MAE loss
# ------------------------------------------
x_test_pred = model.predict(X_aVF_test_3D)
test_mae_loss = np.mean(np.abs(x_test_pred - X_aVF_test_3D), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# Step 10 Compare Reconstruction
# ------------------------------------------
# Checking how the first sequence is learnt
plt.plot(X_aVF_test_3D[3])
plt.plot(x_test_pred[3])
plt.show()


# Step 11 Prepare Synthetic Test Data of Triangle Wave
# ------------------------------------------------------
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt

# Wave 1
freq = 5
amp = 0.5
time = np.linspace(0, 1000, 1000)

signal1 = amp*sg.sawtooth(2*np.pi*freq*time, width=0.5)

plt.plot(time, signal1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Wave 2
freq = 3
amp = 1
time = np.linspace(0, 1000, 1000)

signal2 = amp*sg.sawtooth(2*np.pi*freq*time, width=0.5)

plt.plot(time, signal2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

signal = np.vstack((signal1, signal2))
from numpy import newaxis
signal_3D = signal[:, :, newaxis]
print(signal_3D.shape)
print(signal_3D.shape[0])  # 2
print(signal_3D.shape[1])  # 1000
print(signal_3D.shape[2])  # 1

# Step 12 Get test MAE loss
# ------------------------------------------
x_test_pred1 = model.predict(signal_3D)
test_mae_loss1 = np.mean(np.abs(x_test_pred1 - signal_3D), axis=1)
test_mae_loss1 = test_mae_loss1.reshape((-1))

plt.hist(test_mae_loss1, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss1 > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

plt.plot(signal_3D[0])
plt.plot(x_test_pred1[0])
plt.show()

plt.plot(signal_3D[1])
plt.plot(x_test_pred1[1])
plt.show()







