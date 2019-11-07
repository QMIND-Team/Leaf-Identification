from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
#import cv2

# Place preprosessing code here
# Need x_train, x_test, y_train, y_test

# Model parameters
epochs = 10
batch_size = 34


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
        r'C:\Users\hyper\Downloads\leaf\RGB', # Change this directory
        target_size=(720, 960),
        batch_size=batch_size,
        shuffle=False)



# Basic Neural Net Model for Baseline

input_shape = [720,960,3] # Size of input images in the dataset
num_classes = 40  # Number of leaves to identify 
#history = AccuracyHistory() # For seeing the results of training

# Build Model 

model = Sequential()  # Simple sequential model

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))  # Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) # Pooling Layer
model.add(Conv2D(64, (5, 5), activation='relu')) # Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling Layer
model.add(Flatten())  # Flattening layer
model.add(Dense(1000, activation='relu')) # ??? 1000
model.add(Dense(num_classes, activation='softmax')) # Dense layer

# Compile Model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=batch_size // batch_size,
        epochs=epochs)


#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test),
#          callbacks=[history])


print("Success")

'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

plt.plot(range(1,11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
'''