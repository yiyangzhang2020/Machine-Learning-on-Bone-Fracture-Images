#import required packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torchvision import datasets, transforms
from __future__ import print_function
import os
import time
import datetime
import random
import json
import argparse
import numpy as np
import tensorflow
import tensorflow.keras.backend as K 
from tensorflow.keras.applications import densenet
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import random
import tensorflow
import cv2
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import InceptionV3, MobileNetV2, DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from glob import glob
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D

import matplotlib.image as mpimg

#change this path to to path of where the data is tored
training_path = '/content/drive/MyDrive/Colab Notebooks/MURA-v1.1/train/XR_ELBOW'

def load_path(root_p = training_path):
	
	'''
	load MURA elbow dataset

	'''
	Path = []
	labels = []
    #Read all images
	for root,dirs,files in os.walk(root_p): 
		for name in files:
			if str(name[0]) != '.':
				path_1 = os.path.join(root,name)
				Path.append(path_1)

			#distinguish between postive and negative
			if root.split('_')[-1]=='positive': 
				labels+=[1] 
			elif root.split('_')[-1]=='negative':
				labels+=[0]

# 	labels = np.asarray(labels)
	return Path, labels

X_path, X_label = load_path()



def random_rotation_flip(image,size = 224):
	if random.randint(0,1):
		image = cv2.flip(image,1) 

	if random.randint(0,1):
		angle = random.randint(-30,30)
		M = cv2.getRotationMatrix2D((size/2,size/2),angle,1)
	
		image = cv2.warpAffine(image,M,(size,size))
	return image

def load_image(Path = X_path, size = 224):
	Images = []
	for path in Path:
		try:
			image = cv2.imread(path,cv2.IMREAD_COLOR)
			image = cv2.resize(image,(size,size))
			image = random_rotation_flip(image,size)
			Images.append(image)

		except Exception as e:
			print(str(e))

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images)	
	std = np.std(Images)
	Images = (Images - mean) / std
	
	return Images

X_train = load_image()

print(X_train.shape)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(figsize=(5,5))
c = 5

# plt.plot(X_train)
print(len(X_label))
print(len(X_train))

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

import seaborn as sns

X = np.array(X_train)
y_raw = np.array(X_label)

y = convertToOneHot(y_raw)

# display the distribution of postive and negative
sns.countplot(y_raw)


#print out info for the training set
print("Shape of train images is: ", X.shape)
print("Shape of labels is: ", y.shape)
print(X[0][0].shape)

#Test train split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state=3)

#Test the split of 0's and 1's in training and validation labels
pos_train =0
neg_train = 0
pos_val = 0
neg_val = 0


for i in range(1,len(y_train)):
    if y_train[i][0] == 1:
        pos_train = pos_train + 1
    else:
        neg_train = neg_train + 1
        
for i in range(1,len(y_val)):
    if y_val[i][0] == 1:
        pos_val = pos_val + 1
    else:
        neg_val = neg_val + 1

print("In training:")
print("Positive: " + str(pos_train))
print("Negative: " + str(neg_train))
print("Pos/neg ratio: " + str (pos_train/neg_train))

print("In testing:")
print("Positive: " + str(pos_val))
print("Negative: " + str(neg_val))
print("Pos/neg ratio: " + str (pos_val/neg_val))

print("Shape of Training images is: ", X_train.shape)
print("Shape of Validation images is: ", X_val.shape)
print("Shape of Training labels is: ", y_train.shape)
print("Shape of Validation labels is: ", y_val.shape)

ntrain = len(X_train)
print("Number of training images: ", ntrain)
nval = len(X_val)
print("Number of validation images: ",nval)

batch_size = 20

IMG_SIZE = (224,224,3)
inp = Input(IMG_SIZE)

densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp, input_shape=IMG_SIZE, pooling='max')
input_layer=densenet.inputs

x = densenet.output
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax', name='predictions')(x)

# Create model
feature_model = models.Model(input_layer, preds)

feature_model.trainable = True

opt = SGD(learning_rate=0.001, momentum=0.0, nesterov = True, name="SGD")


for layer in feature_model.layers[:-60]:
    layer.trainable = False
for layer in feature_model.layers[-60:]:
    layer.trainable = True

feature_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range = 30,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  shear_range=0.3,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  data_format='channels_last')

val_datagen = ImageDataGenerator(rescale = 1./255)

# Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
modelcheckpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath='keras_mobilenetv2_model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True, monitor='val_acc', mode='max')

log_dir = "./log/mobilenetv2/50 Layers Trained/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, write_graph=True, update_freq='batch', embeddings_freq=0, embeddings_metadata=None)

reducelr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, mode='auto', cooldown=0)
progbar = tensorflow.keras.callbacks.ProgbarLogger(count_mode='samples')

my_callbacks = [
    earlystop,
    modelcheckpoint,
    tensorboard,
    reducelr,
    progbar,
]

EPOCHS = 50

feature_model.summary()

# Train the model
history = feature_model.fit_generator(train_generator,
                                    steps_per_epoch = ntrain // batch_size,
                                    epochs = EPOCHS,
                                    validation_data = val_generator,
                                    verbose=1,
                                    callbacks=my_callbacks)

# Save the model
feature_model.save_weights('denenet121_forearm_fracture_detection_weights.h5')
feature_model.save('denenet121_forearm_fracture_detection_model.h5')


model_json = feature_model.to_json()
with open("denenet121_forearm_fracture_detection_model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")


# Plot train and validation accuracy curves
# get the details from history object

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#train and val accuracy
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()


plt.figure()
plt.savefig('/Training_Validation_Accuracy.png')

# Train and val loss
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()


plt.show()
plt.savefig('/Training_Validation_Loss.png')
