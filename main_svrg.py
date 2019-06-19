import json
import keras
import argparse

import matplotlib.pyplot as plt

from cifar10 import Cifar10
from importance_sampling.training import SVRG


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--epochs', '-e', default=25, type=int,
                        help='number of epochs')
    parser.add_argument('--learning-rate', '-l', default=0.01,
                        type=float, help='learning rate')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='use this flag to plot the training history')
    parser.add_argument('--save', '-s', default=None,
                        help='Specify a path to save training history as a JSON file')
    args = parser.parse_args()
    return args


def main(batch_size, epochs, learning_rate, plot):
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data(
        y_test_to_categorical=True
    )
    model = Cifar10.load_model()
    optimizer = keras.optimizers.SGD(
        lr=learning_rate,
        momentum=0.0,
        decay=0.0,
        nesterov=False
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    svrg_model = SVRG(model, B=0, B_over_b=len(x_train) // batch_size)

    history = {'training': {}, 'test': {}}
    training_history = svrg_model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )
    history['training']['accuracy'] = [
        float(i) for i in training_history.history['accuracy']
    ]
    history['test']['accuracy'] = [
        float(i) for i in training_history.history['val_accuracy']
    ]

    # summarize history for accuracy
    if plot:
        plt.plot(history['training']['accuracy'])
        plt.plot(history['test']['accuracy'])
        plt.xlabel('epoch')
        plt.legend(['training', 'test'], loc='upper left')
        plt.show()

    return history


if __name__ == '__main__':
    args = parse_arguments()
    history = main(
        batch_size=args.batch,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        plot=args.plot
    )
    if args.save:
        with open(args.save, 'w') as fobj:
            json.dump(history, fobj)
