import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Cifar10(object):

    NUM_CLASSES = 10

    @classmethod
    def load_data(cls, y_test_to_categorical=False):
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, Cifar10.NUM_CLASSES)
        if y_test_to_categorical:
            y_test = keras.utils.to_categorical(y_test, Cifar10.NUM_CLASSES)

        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def load_model(cls):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        return model
