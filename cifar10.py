import keras
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from trish import TRish

batch_size = 128  # 32 for SGD
num_classes = 10
epochs = 25


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3))
    )
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
optimizer = TRish(alpha=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
history = {
    'training': {
        'accuracy': []
    },
    'test': {
        'accuracy': []
    }
}
for epoch in range(epochs):
    epoch_result = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=1
    )
    history['training']['accuracy'].append(epoch_result.history['acc'][0])
    y_hat = model.predict_classes(x_test)
    test_accuracy = len(y_hat[y_hat == y_test.ravel()]) / len(y_test)
    print('test acc:', test_accuracy)
    history['test']['accuracy'].append(test_accuracy)

# summarize history for accuracy
plt.plot(history['training']['accuracy'])
plt.plot(history['test']['accuracy'])
plt.xlabel('epoch')
plt.legend(['training', 'test'], loc='upper left')
plt.show()

print(history)
