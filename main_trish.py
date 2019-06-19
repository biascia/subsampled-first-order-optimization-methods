import json
import argparse

import matplotlib.pyplot as plt

from trish import TRish
from cifar10 import Cifar10
from training import train


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--epochs', '-e', default=25, type=int,
                        help='number of epochs')
    parser.add_argument('--alpha', '-a', default=0.1,
                        type=float, help='alpha')
    parser.add_argument('--gamma-1', '-g1', default=1,
                        type=float, help='gamma 1')
    parser.add_argument('--gamma-2', '-g2', default=0.001,
                        type=float, help='gamma 2')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='use this flag to plot the training history')
    parser.add_argument('--save', '-s', default=None,
                        help='Specify a path to save training history as a JSON file')
    args = parser.parse_args()
    return args


def main(batch_size, epochs, alpha, gamma_1, gamma_2, plot):
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data()
    model = Cifar10.load_model()
    optimizer = TRish(
        alpha=alpha,
        gamma_1=gamma_1,
        gamma_2=gamma_2
    )
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
        alpha=args.alpha,
        gamma_1=args.gamma_1,
        gamma_2=args.gamma_2,
        plot=args.plot
    )
    if args.save:
        with open(args.save, 'w') as fobj:
            json.dump(history, fobj)
    """
    alphas = [0.01, 0.1, 1]
    gamma_1s = [0.2, 1, 5]
    gamma_2s = [0.001, 0.01, 0.1]
    batches = [32, 64, 128]
    epochs = 25

    results = []
    for batch in batches:
        for alpha in alphas:
            for gamma_1 in gamma_1s:
                for gamma_2 in gamma_2s:
                    history = main(
                        batch_size=batch,
                        epochs=epochs,
                        alpha=alpha,
                        gamma_1=gamma_1,
                        gamma_2=gamma_2,
                        plot=False
                    )
                    result = {
                        'params': {
                            'batch': batch,
                            'alpha': alpha,
                            'gamma_1': gamma_1,
                            'gamma_2': gamma_2
                        },
                        'history': history
                    }
                    results.append(result)
    import json
    json.dump(results, open('results/trish.json', 'w'))
    """