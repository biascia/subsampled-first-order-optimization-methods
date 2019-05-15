from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K


class TRish(Optimizer):
    """TRish optimizer.
    """

    def __init__(self, alpha=1, gamma_1=1.051, gamma_2=0.0089, **kwargs):
        if gamma_1 <= 0 or gamma_2 <= 0 or gamma_2 >= gamma_1:
            raise ValueError('Expecting gamma_1 > gamma_2 > 0')
        if alpha <= 0:
            raise ValueError('Expecting alpha > 0')
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.alpha = K.variable(alpha, name='alpha')
            self.gamma_1 = K.variable(gamma_1, name='gamma_1')
            self.gamma_2 = K.variable(gamma_2, name='gamma_2')
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]
        grads = self.get_gradients(loss, params)
        g_norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))

        alpha = self.alpha
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2

        for p, g in zip(params, grads):
            new_p = K.switch(
                K.less(g_norm, 1 / gamma_1),
                p - gamma_1 * alpha * g,
                # p - alpha * g / g_norm
                K.switch(
                    K.less_equal(g_norm, 1 / gamma_2),
                    p - alpha * g / g_norm,
                    p - gamma_2 * alpha * g
                )
            )
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'gamma_1': float(K.get_value(self.gamma_1)),
                  'gamma_2': float(K.get_value(self.gamma_2))}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))