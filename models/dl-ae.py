# Deep Learning - Autoencoder
#-----------------------------------------------

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os


# Part 1 Data Preprocessing
#-------------------------------------------------------
# 1a Importing the dataset

os.getcwd()
os.chdir("D:\\Study\\All_datasets")
movies = pd.read_csv("ae-ml-1m\\movies.dat", sep='::', header=None, engine='python',encoding='latin-1')
users = pd.read_csv("ae-ml-1m\\users.dat", sep='::', header=None, engine='python',encoding='latin-1')
ratings = pd.read_csv("ae-ml-1m\\ratings.dat", sep='::', header=None, engine='python',encoding='latin-1')

#1b Prepare training set and test set

training_set = pd.read_csv("ae-ml-100k\\ml-100k\\u1.base", delimiter = '\t')
# convert to array
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv("ae-ml-100k\\ml-100k\\u1.test", delimiter = '\t')
test_set = np.array(test_set, dtype='int')

# Getting no of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Change training set format to matrix where lines users, columns movies and cells rating
# We change to such structure which will fit to structure of autoencoders

# Observations (users) in rows and features (movies) in columns and cells are
# ratings
# In ecg instead of fixed movies we have fixed timestamp n ratings are diff
# ecg signal values for each patient so kind of time series data for us
# and time stamps we are putting as features
# 983 users and 1682 movies, put 0 whichever movie user did not rate

# Converting data into an array with users in lines and movies in columns
def convert(data):
    new_data = []  # whole lists
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # ratings start from 0 and id_movies from 1 so
        new_data.append(list(ratings)) # to start id_movies also from index 0 we subtract 1
    return new_data

# Call function create to create training and test set into special structure
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# Training_set is a list of lists & FloatTensor class expects a list of lists
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Part 2 Building the Autoencoder
# -----------------------------------------------------
# Create a child class # SAE means we have many hidden layers so will have
# several encodings of the input vector features
# Parent class Module, child class SAE so process of inheritance comes into picture
# Library -> Module -> Class -> Object
# nn is a module and Module is a class in nn module

# Inheritance
# In this class 2 functions 1st function init to define the architecture of
# neural network and 2nd function forward will do action of encoding & decoding &
# also apply the activation function b/w the full connections.

class SAE(nn.Module):
    # init is a function, self is the object of SAE class so AE
    def _init_(self, ):
        super(SAE, self)._init_() # This will make sure we get all the
        # inherited classes and methods of parent class Module.
        # With this no 20 we r trying to detect 20 features-1st hidden layer
        self.fc1 = nn.Linear(nb_movies, 20 )
        # This will detect 10 more features but based on previous features
        # that were detected
        self.fc2 = nn.Linear(20, 10)
        # Now no more encoding, starting to decode
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        # 5 objects out of which 1 object of sigmoid class and 4 objects
        # of linear class
    def forward(self, x):
        x = self.activation(self.fc1(x)) # encoding
        x = self.activation(self.fc2(x)) # encoding
        x = self.activation(self.fc3(x)) # decoding
        x = self.fc4(x) # decoding
        # Predict, calculate loss and then adjust weights to reduce this loss
        return x  # vector of predicted ratings

# Create an object of this class (use non capital letters for object)
# the class is the instructions to build the autoencoder
sae = SAE()  # this is an instance/object of class hence an ae
criterion = nn.MSELoss()
# optimizer applies SGD to update the weights inorder to reduce error at
# each epoch
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
# decay is used to reduce the learning rate after every few epochs and
# that's inorder to regulate the convergence


# Part 3 Training the Stacked Autoencoder
# ----------------------------------------------------
# At each epoch the weights can be updated
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    # compute loss at each epoch & see if it is decreasing over the epochs
    # optimizer that will apply SGD to update the weights & lead to convergence
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)

        # creating batch of 1 input vector cz it does not accept 1D vector
        target = input.clone()
        # memory optimization, consider only users who rated
        if torch.sum(target.data > 0) > 0:
            # get op vector of predicted ratings
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # Backward method for loss. tells in which direction we need to
            # update the wts. Inc wt or decrease wt?
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            # Difference b/w backward(decides DIRECTION of wts) and optimizer(decides AMOUNT of wts)
            # backward decides the direction to which the wts will be updated
            # that is will they be increased or decreased & optimizer step
            # decides the intensity/amount by which the wts will be updated.
            optimizer.step()
    print('epoch: '+str(epoch)+ 'loss: '+str(train_loss/s))

# Part 3 Testing the Stacked Autoencoder
# -------------------------------------------------

test_loss = 0
s = 0.
# compute loss at each epoch & see if it is decreasing over the epochs
# optimizer that will apply SGD to update the weights & lead to convergence
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)

    # creating batch of 1 input vector cz it does not accept 1D vector
    target = Variable(test_set[id_user]).unsqueeze(0)

    # memory optimization, consider only users who rated
    if torch.sum(target.data > 0) > 0:
        # get op vector of predicted ratings
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        # Backward method not req as related to training
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))