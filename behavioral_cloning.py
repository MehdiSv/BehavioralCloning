import json

import math
import h5py
import numpy as np
import pandas
import cv2
from PIL import Image, ImageTk
from keras.layers import Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.ndimage import imread
from sklearn.utils import shuffle
from collections import defaultdict

img_width = 64
img_height = 64

batch_size = 256
epochs = 1

global data, y_train

stats = defaultdict(float)

# This method transform the image to HSV and randomizes the V value to change the brightness.
# This will help our network. It will generate more easily and won't be affected by change in light conditions.
def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def X_generator():
    global data, stats
    
    data = shuffle(data)
    
    images = np.zeros((batch_size, img_width, img_height, 3))
    steerings = np.zeros(batch_size)    

    while True:
        for i_batch in range(batch_size):
            line_number = np.random.randint(len(data))
            data_to_process = data.iloc[[line_number]].reset_index()                    
            # Get center left or right image
            rand = np.random.randint(3)
            if (rand == 0):
                path_file = data_to_process['left'][0].strip()
                shift_ang = .3
            elif (rand == 1):
                path_file = data_to_process['right'][0].strip()
                shift_ang = -.3
            else:
                path_file = data_to_process['center'][0].strip()
                shift_ang = 0.
            steering = data_to_process['steering'][0] + shift_ang

            # Keep only the relevant part of the road in the image and normalize
            image = cv2.imread(path_file.strip())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = randomize_brightness(image)
            shape = image.shape
            image = image[math.floor(shape[0] / 5.):shape[0] - 25, 0:shape[1]]
            image = cv2.resize(image, (img_height, img_width), interpolation=cv2.INTER_AREA)
            image = image / 255.
            image = np.array(image)

            # Flip 50% of the time
            if np.random.randint(2) == 0:
                image = cv2.flip(image, 1)
                steering = -steering

            images[i_batch] = image
            steerings[i_batch] = steering
            stats[round(steering, 1)] += 1
        yield images, steerings


def load_data():
    global data, y_train

    data = pandas.read_csv('data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])    
    y_train = data['steering'].values

# This method describes the network architecture
def SmallNetwork(input_shape):
    model = Sequential()
    # 2 CNNs blocks comprised of 32 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    # Maxpooling
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 2 CNNs blocks comprised of 64 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    # Maxpooling + Dropout to avoid overfitting
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))    

    # 2 CNNs blocks comprised of 128 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    # Last Maxpooling. We went from an image (64, 64, 3), to an array of shape (8, 8, 128)
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))    

    # Fully connected layers part.
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='elu'))
    # Dropout here to avoid overfitting
    model.add(Dropout(0.5))    
    model.add(Dense(64, activation='elu'))
    # Last Dropout to avoid overfitting
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu'))    
    model.add(Dense(1))

    return model

if __name__ == '__main__':
    global y_train, stats

    load_data()

    input_shape = (3, img_width, img_height)

    model = SmallNetwork(input_shape)
    opt = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=opt)
    for i in range(5):
        train_generator = X_generator()
        train_validator = X_generator()
        model.fit_generator(train_generator, samples_per_epoch=len(y_train), nb_epoch=epochs, validation_data=train_validator, nb_val_samples=len(y_train) / 6)


    with open('model.json', 'w') as fd:
        json.dump(model.to_json(), fd)

    model.save_weights('model.h5')
    print(stats)
