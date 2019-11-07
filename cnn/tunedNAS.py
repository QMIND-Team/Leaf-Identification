from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetLarge
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
import numpy as np
#import cv2

# Model parameters
epochs = 10

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = datagen.flow_from_directory(
        r'/Users/lukaantonyshyn/Desktop/qmind/Leaf-Identification/Leaves', #Make this generalized at some point
        target_size=(331, 331),
        shuffle=False)

input_shape = [331,331,3] # Size of input images in the dataset
num_classes = 32  # Number of leaves to identify 

# Build Model 
from keras import backend as K
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

base_model = NASNetLarge(include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit_generator(
        train_generator,
        epochs=epochs)

for layer in model.layers[689:]:
   layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(
        train_generator,
        epochs=epochs)

print('Success!')
