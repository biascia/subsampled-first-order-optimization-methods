import argparse

import matplotlib.pyplot as plt

from cifar10 import Cifar10
from training import train
from saga import SAGA


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--epochs', '-e', default=25, type=int,
                        help='number of epochs')
    parser.add_argument('--gamma', '-g', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='use this flag to plot the training history')
    args = parser.parse_args()
    return args


def main(batch_size, epochs, gamma, plot):
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data()
    model = Cifar10.load_model()
    optimizer = SAGA(gamma=gamma)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = train(
        model=model,
        train_set=(x_train, y_train),
        test_set=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs
    )

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
        gamma=args.gamma,
        plot=args.plot
    )
