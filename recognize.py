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
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load training data
prestring0 = 'trainingData/0/img' #x.png for open hand
prestring1 = 'trainingData/1/img' #x.png for closed hand
#prestring2 = 'trainingData/2/img' #x.png
#prestring3 = 'trainingData/3/img' #x.png
postString = '.png'

imNum0 = len(fnmatch.filter(os.listdir('trainingData/0'), '*.png'))
imNum1 = len(fnmatch.filter(os.listdir('trainingData/0'), '*.png'))



shape0 = (imNum0, 3, 32, 32)
shape1 = (imNum1, 3, 32, 32)
openHand = np.zeros(shape)			#np.array for open hand images
closedHand = np.zeros(shape)		#np.array for closed hand images

for x in range (0 to imNum0): #for each image in trainingData/0/*
	fname = prestring0 + x + postString
	if os.path.exists(fname):
		temp = cv2.imread(fname)
		for rgb in range (0 to 3):
			openHand[x][rgb] = temp[:,:][rgb] #load it into openHand
	else:
		imNum0++
	



# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))\