# Deep Learning - ANN
#-----------------------------------------------

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os

import tensorflow.keras.models

# Part 1 Data Preprocessing
# Part 2 Building the ANN
# Part 3 Training the ANN
# Part 4 Making predictions and evaluating the model

# Part 1 Data Preprocessing
#-------------------------------------------------------
# 1a Importing the dataset

os.getcwd()
os.chdir("/Users/sylvia/Desktop/Sylvia/Datasets")
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values

# 1b Encoding categorical variables
# Label encoding gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

# One hot encoding for geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

# 1c Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# 1d Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 Building the ANN
#-------------------------------------------------
# 2a Initialize ANN as a sequence of layers.
# ann as object/instance of class sequential
ann = tensorflow.keras.models.Sequential()

# 2b Add input layer and 1st hidden layer
# add is a method of sequential class and since ann
# is an object of Sequential class it can access the method
# from layers module we call dense class
# This is a shallow NN now
# These 6 neurons for hidden layer, for i/p it will take automatically.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2c Add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2d Add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part 3 Train the ANN
#--------------------------------------------------------
# We only build the artificial brain in Part 2 but now we
# make it smart by training it on the data.

# 3a Compiling the ANN with optimizer, loss function and a metric
# Optimizer will update the weights through Stochastic Gradient Descent
# Loss function is a way to compute the difference b/w pred's & real result
# metrics - with which u want to evaluate your ANN during training.
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3b Training ANN on whole training dataset over certain no of epochs
ann.fit(X_train, y_train, batch_size=32, epochs=50)

# Part 4 Predicting and evaluating the model
#-------------------------------------------------
# Predicting a single observation
# Input of predict is a 2D array [[]]
print(ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# Sigmoid activation will return ans in form of probability - 0.5 threshold to say yes or no

# Predicting test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# Making confusion matrix and check accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test,y_pred)
print(ac)

