import random

from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K


class SAGA(Optimizer):
    """SAGA optimizer - no regularization
    """

    def __init__(self, gamma=0.01, seed=1926, **kwargs):
        with K.name_scope(self.__class__.__name__):
            random.seed(seed)
            self.table_initialized = False
            self.seed = seed
            self.gamma = K.variable(gamma, name='gamma')
            self.seed = K.variable(seed, dtype='int64', name='seed')
            self.table = K.floatx()
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]
        grads = self.get_gradients(loss, params)
        if not self.table_initialized:
            # initialize table
            self.table = grads
            self.table_initialized = True
        j = random.choice(range(len(params)))
        new_grad = grads[j]
        old_grad = self.table[j]

        for p, i in zip(params, range(len(params))):
            new_p = p - self.gamma * self.table[i] / len(params)
            if i == j:
                new_p = new_p - self.gamma * new_grad + self.gamma * old_grad
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'gamma': float(K.get_value(self.gamma)), 'seed': self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
