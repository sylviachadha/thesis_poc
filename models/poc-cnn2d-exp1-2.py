# Aim - ptb-xl - deep learning CNN Classification
# Normal vs IMI (subclass with max anomaly samples 1250)
# --------------------------------------------------#
# STEP 1 - Load data   (# AS PAPER)
# --------------------------------------------------#
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# path = '/Users/sylvia/Desktop/Thesis/Dataset/ptb-xl/'
path = 'D:\\Study\\All_datasets\\ptb-xl\\'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
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


def aggregate_diagnostic1(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))


# Apply diagnostic subclass
Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_diagnostic1)

# ---------------------------------------------------------------#
# STEP 2 - Extracting Class of Anomaly out of 71 classes in Y
# ---------------------------------------------------------------#
# Once original X and y are loaded u shorten the y to NORMAL
# and ANOMALY class u want and then load corresponding X again

# Add clean class
Y_short = Y.copy()
Y_short['clean_class'] = Y_short['diagnostic_subclass'].apply(' '.join)
#Y_short['clean_class'] = Y_short['diagnostic_superclass'].apply(' '.join)

# Get unique value counts from a column of a dataframe to
# choose which abnormal class has the most data
uniqueValues = Y_short['clean_class'].unique()
uniqueValues_count = Y_short['clean_class'].nunique()
from collections import Counter
all_unique_values = Counter(Y_short['clean_class'])
print(all_unique_values)


# Make a short df with anomaly and normal class
Y_anomaly = Y_short[Y_short["clean_class"] == "IMI"]
Y_normal = Y_short[Y_short["clean_class"] == "NORM"]
frames = [Y_normal, Y_anomaly]
Y_short = pd.concat(frames)

# Check counts of normal and anomaly class
value_counts = Y_short['clean_class'].value_counts()
print(value_counts)

# Since Filtering and value counts done, remove the clean class
del Y_short['clean_class']

# Load corresponding X as per Y_short
X_short = load_raw_data(Y_short, sampling_rate, path)


# -----------------------------------------------------------------#
# STEP 3 - Train/Test Split   (# AS PAPER)
# -----------------------------------------------------------------#
# 10 fold stratified sampling
test_fold = 10
# Train
X_train_short = X_short[np.where(Y_short.strat_fold != test_fold)]
y_train_short = Y_short[(Y_short.strat_fold != test_fold)].diagnostic_subclass
# Test
X_test_short = X_short[np.where(Y_short.strat_fold == test_fold)]
y_test_short = Y_short[Y_short.strat_fold == test_fold].diagnostic_subclass

# Change to univariate time series (Only Lead V1]
# AMI - V1 and V2

# -----------------------------------------------------------------#
# STEP 4 - Extracting Channel / Lead out of 12 leads
# Preparing X for the model
# -----------------------------------------------------------------#
# Change X from 3D to 2D with only Lead V5 Signal for each patient

print(X_train_short.shape)
print(X_test_short.shape)
# Select only Lead V5 from X which is column 10
# Leads (I, II, III, AVL, AVR, AVF, V1, ..., V6)

X_train_short_V1 = X_train_short[:, :, 6]
X_test_short_V1 = X_test_short[:, :, 6]


# -----------------------------------------------------------------#
# STEP 5 - Changing ['NORM'] to NORM and further to 0 (encoding)
# Preparing y for the model
# -----------------------------------------------------------------#
y_train_short.index.name
y_train_short = y_train_short.apply(' '.join)
y_test_short = y_test_short.apply(' '.join)

# Replace NORM with 0 and Anomaly Class with 1
y_train_short_num = y_train_short.replace(to_replace={"NORM": 0, "IMI": 1})
y_test_short_num = y_test_short.replace(to_replace={"NORM": 0, "IMI": 1})

testlabel_counts = y_test_short_num.value_counts()
print(testlabel_counts)
trainlabel_counts = y_train_short_num.value_counts()
print(trainlabel_counts)

# Plot all Normal and abnormal train patterns and save them in
# separate folders normal_ecg and abnormal_ecg

X_train_short_V1   # ndarray # 0 8170 (0-8169) and 1 1111(8170-9280)
X_train_total = pd.DataFrame(X_train_short_V1)
X_train_abnormal = X_train_total[8170:9281]
X_train_normal = X_train_total[0:8170]

# Single plot
X_train_normal.iloc[100].plot()
plt.show()
X_train_abnormal.iloc[200].plot()
plt.show()

# Save images to folder
# All train normal plots
from PIL import Image
#images_list = []
for x in range(8170):
    f = plt.figure()
    X_train_normal.iloc[x].plot()
    plt.show()
    f.savefig("D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\RGB\\normal_ecg\\normal_ecg" + str(x))
    image1 = Image.open(r"D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\RGB\\normal_ecg\\normal_ecg" + str(x) + '.png')
    im1 = image1.convert('LA')
    im1.save("D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\Grayscale\\normal_ecg\\normal_ecg" + str(x)+'.png')

    #images_list.append(image1)
    #image1.save(r'/Users/sylvia/Desktop/Thesis/Documents/train_ecg/normal_train_ecg/normal_images.pdf',save_all=True, append_images=images_list)

# Save images to folder
# All train abnormal plots
from PIL import Image

#images_list = []
for x in range(1111):
    f = plt.figure()
    X_train_abnormal.iloc[x].plot()
    plt.show()
    f.savefig("D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\RGB\\abnormal_ecg\\abnormal_ecg" + str(x))
    image1 = Image.open(
        r"D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\RGB\\abnormal_ecg\\abnormal_ecg" + str(x) + '.png')
    im1 = image1.convert('LA')
    im1.save("D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\Grayscale\\abnormal_ecg\\abnormal_ecg" + str(x) + '.png')


# Plot all Normal and abnormal TEST patterns and save them in
# separate folders test_normal_ecg and test_abnormal_ecg

X_test_short_V1   # ndarray # 0 are 913 and 1 are 256
X_test_total = pd.DataFrame(X_test_short_V1)
X_test_abnormal = X_test_total[913:1052]
X_test_normal = X_test_total[0:913]

# Single plot
X_test_normal.iloc[100].plot()
plt.show()
X_test_abnormal.iloc[100].plot()
plt.show()

# Save images to folder
# All train normal plots
from PIL import Image
#images_list = []
for x in range(913):
    f = plt.figure()
    X_test_normal.iloc[x].plot()
    plt.show()
    f.savefig("D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\RGB\\normal_ecg\\normal_ecg" + str(x))
    image1 = Image.open(
        r"D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\RGB\\normal_ecg\\normal_ecg" + str(x) + '.png')
    im1 = image1.convert('LA')
    im1.save("D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\Grayscale\\normal_ecg\\normal_ecg" + str(x) + '.png')

# Save images to folder
# All test abnormal plots
from PIL import Image

#images_list = []
for x in range(139):
    f = plt.figure()
    X_test_abnormal.iloc[x].plot()
    plt.show()
    f.savefig("D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\RGB\\abnormal_ecg\\abnormal_ecg" + str(x))
    image1 = Image.open(
        r"D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\RGB\\abnormal_ecg\\abnormal_ecg" + str(x) + '.png')
    im1 = image1.convert('LA')
    im1.save("D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\Grayscale\\abnormal_ecg\\abnormal_ecg" + str(x) + '.png')

# Part 1 Data Preprocessing
# Part 2 Building the CNN
# Part 3 Training the CNN
# Part 4 Making predictions and evaluating the model

# Part 1 Data Preprocessing
#-----------------------------------------------

# 1a Preprocessing the Training set
# Apply transformations on training set to avoid overfitting
# Geometrical transformations - zooms / rotations on images / horizontal flips
# Image Augmentation - by applying these transformations we will get new images
# Augment the variety / diversity of training set images

import pandas as pd
import numpy as np
import sklearn
import wfdb
import ast
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,# rescale applies feature scaling to each & every pixels
    # dividing their value by 255 cz each pixel takes a value b/w 0 and 255
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
)

train_set = train_datagen.flow_from_directory(
    "D:\\Study\\All_datasets\\ptb-xl\\train_ecg\\Grayscale",
    target_size=(64,64), # size of image when they will be fed to cnn
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

# 1b Preprocessing the Test set

test_datagen = ImageDataGenerator(
    rescale=1./255,
    )

test_set = test_datagen.flow_from_directory(
    "D:\\Study\\All_datasets\\ptb-xl\\test_ecg\\Grayscale",
    target_size=(64,64), # size of image when they will be fed to cnn
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)


# Part 2 Building the CNN
#-----------------------------------------------

# 2a Initializing the CNN
# CNN is a sequence of layers so we initialize with sequential class as
# opposed to computational graph

cnn = tf.keras.models.Sequential()

# 2b Add convolutional layer
# filters is no of feature detectors u want to apply to your images
# kernel size is size of feature detector if u choose 3 size will be 3*3
cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation='relu',
                               input_shape=[64,64,1]))
# input_shape=[64,64,3] if coloured image, input_shape=[64,64,1] if black/white

# 2c Add Pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2d Add a second convolutional layer with max pooling applied
cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=2,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2e Flattening
# All convolutions & pooling into 1d vector which will be i/p to future fully connected NN

cnn.add(tf.keras.layers.Flatten())

# 2f Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# 2g Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.summary()
# In machine learning, we use sigmoid to map predictions to probabilities.

# Part 3 Training the CNN
#-----------------------------------------------

# Make this artificial brain with some eyes pretty smart to recognise cats
# and dogs in images.

# 3a Compiling the cnn
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3b Training CNN on Training set and evaluating at same time on test set (only in CNN)

cnn.fit(x=train_set, validation_data=test_set, epochs=35)

# Hyperparameters in CNN
# Epochs, Optimizer

# Part 4 Making single Prediction
#-----------------------------------------------------

# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('D:\\Study\\All_datasets\\ptb-xl\\single_prediction/normal_ecg15.png',target_size=(64,64),color_mode="grayscale")
# test_image = image.img_to_array(test_image)
# # since test_image of batch dimension
# test_image = np.expand_dims(test_image, axis=0)
#
# result = cnn.predict(test_image/255.0) # will return 0 and 1
# print(train_set.class_indices)
# # accessing batch and then single element of batch
# if result[0][0] > 0.5:
#     prediction = 'normal_ecg'
# else:
#     prediction = 'abnormal_ecg'
#
# print(prediction)

# Part 5 Make all predictions and evaluate the model
#-----------------------------------------------------

# https://stackoverflow.com/questions/50825936/confusion-matrix-on-images-in-cnn-keras
# test_generator = ImageDataGenerator()
# test_data_generator = test_generator.flow_from_directory(
#     test_data_path, # Put your path here
#      target_size=(img_width, img_height),
#     batch_size=32,
#     shuffle=False)


test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

predictions = cnn.predict_generator(test_set, steps=test_steps_per_epoch)
# Get most likely class
# Argmax is most commonly used in machine learning for finding the class
# with the largest predicted probability.
# predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = tf.greater(predictions, 0.8)


true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

# Classification Report
from sklearn.metrics import classification_report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)