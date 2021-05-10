# Deep Learning - CNN
#-----------------------------------------------

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf._version_

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


train_datagen = ImageDataGenerator(
    rescale=1./255,# rescale applies feature scaling to each & every pixels
    # dividing their value by 255 cz each pixel takes a value b/w 0 and 255
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_set = train_datagen.flow_from_directory(
    "D:\\Study\\All_datasets\\cnn_dataset\\training_set",
    target_size=(64,64), # size of image when they will be fed to cnn
    batch_size=32,
    class_mode='binary'
)

# 1b Preprocessing the Test set

test_datagen = ImageDataGenerator(
    rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'D:\\Study\\All_datasets\\cnn_dataset\\test_set',
    target_size=(64,64), # size of image when they will be fed to cnn
    batch_size=32,
    class_mode='binary'
)


# Part 2 Building the CNN
#-----------------------------------------------

# 2a Initializing the CNN
# CNN is a sequence of layers so we initialize with sequential class as
# opposed to computational graph

cnn = tf.keras.models.Sequential()

# 2b Add convolutional layer
# filters is no of feature detectors u want to apply to your images
# kernel size is size of feature detector if u choose 3 size will be 3*3
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',
                               input_shape=[64,64,3]))
# input_shape=[64,64,3] if coloured image, input_shape=[64,64,1] if black/white

# 2c Add Pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2d Add a second convolutional layer with max pooling applied
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2e Flattening
# All convolutions & pooling into 1d vector which will be i/p to future fully connected NN

cnn.add(tf.keras.layers.Flatten())

# 2f Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# 2g Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part 3 Training the CNN
#-----------------------------------------------

# Make this artificial brain with some eyes pretty smart to recognise cats
# and dogs in images.

# 3a Compiling the cnn
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3b Training CNN on Training set and evaluating at same time on test set (only in CNN)

cnn.fit(x=train_set, validation_data=test_set, epochs=25)

# Part 4 Making predictions and evaluating the model
#-----------------------------------------------------

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:\\Study\\All_datasets\\cnn_dataset\\single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
# since test_image of batch dimension
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image/255.0) # will return 0 and 1
print(train_set.class_indices)
# accessing batch and then single element of batch
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)