import cv2
import numpy as np
from os import path
from os import listdir
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# (X_train1, y_train1), (X_test1, y_test1) = cifar10.load_data()
# print(X_train1.shape)
# print(y_train1.shape)


K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load training data
prestring0 = 'trainingData/0/img' #x.png for open hand
prestring1 = 'trainingData/1/img' #x.png for closed hand
#prestring2 = 'trainingData/2/img' #x.png
#prestring3 = 'trainingData/3/img' #x.png
postString = '.png'

# imNum0 = len(fnmatch.filter(os.listdir('trainingData/0'), '*.png'))
# imNum1 = len(fnmatch.filter(os.listdir('trainingData/0'), '*.png'))
imNum0 = 10000
imNum1 = 10000

shape0 = (imNum0, 32, 32, 3)

num_classes = 2
X_train = np.ndarray(shape0)
y_train = np.ndarray((imNum0))

count = 0
for x in range(0, imNum0): #for each image in trainingData/0/*
  fname = prestring0 + str(x) + postString
  if path.exists(fname):
    temp = cv2.imread(fname)
    X_train[count] = temp
    y_train[count] = 0

    count += 1
    # for rgb in range (0, 3):
    # openHand[x][rgb] = temp[:,:][rgb] #load it into openHand

print(X_train.shape)

for x in range(0, imNum1): #for each image in trainingData/0/*
  fname = prestring1 + str(x) + postString
  if path.exists(fname):
    temp = cv2.imread(fname)
    X_train[count] = temp
    y_train[count] = 1

    count += 1
    # for rgb in range (0, 3):
    # openHand[x][rgb] = temp[:,:][rgb] #load it into openHand

X_train = X_train[:count]



# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

print(X_train.shape)


# Fit the model
model.fit(X_train, y_train, validation_data=(X_train, y_train), nb_epoch=epochs, batch_size=32)
"""
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
