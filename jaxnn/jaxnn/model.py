from functools import partial

from jaxnn.utils import random_key
from . import nn
import jax
from jax import numpy as jnp


class Model:
    def __init__(self, layers) -> None:
        self._init_fn, self._call_fn = nn.net(layers=layers)
        self.state = None

    def fit(self,
            data_loader,
            optimizer,
            loss_fn,
            epochs=1,
            metrics=[],
            rng=random_key(),
            *args,
            **kwargs):
        self.losses = []
        self.valid_losses = []
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        for (epoch, (train_iter,
                     (valid_x, valid_y))) in zip(range(epochs), data_loader):
            for (x, y) in train_iter:
                loss = self._step(x, y, rng)
            self._record(epoch, loss, valid_x, valid_y, metrics)

    def _record(self, epoch, loss, valid_x, valid_y, metrics):
        self.losses.append(loss)
        valid_y_hat = self.predict(valid_x)
        valid_loss = self.loss_fn(valid_y_hat, valid_y)
        self.valid_losses.append(valid_loss)
        extra = ''
        for metric in metrics:
            metric_fn = self.__getattribute__(metric)
            metric_val = metric_fn(valid_y_hat, valid_y)
            extra += f'valid {metric}: {metric_val:.3f}'
        comp, uncomp = self._calc_complete(epoch + 1, self.epochs)
        print(
            f'epoch: [{epoch + 1:>3}\\{self.epochs} {comp}{uncomp} loss: {loss:<.3f} valid loss: {valid_loss:<.3f} {extra}'
        )

    @staticmethod
    def _calc_complete(epoch, epochs):
        per = epoch / epochs
        complete = int(per * 20)
        uncomplete = 20 - complete
        return complete * '#', uncomplete * '-'

    def _step(self, x, y, rng):
        if self.state is None:
            self.state = self._init_fn(x.shape[1:], rng)
        loss, self.state = self._update(state=self.state, x=x, y=y)
        return loss

    def initialize(self, feature_shape, rng=random_key()):
        self.state = self._init_fn(feature_shape, rng)

    @partial(jax.jit, static_argnums=0)
    def _update(self, state, x, y):
        loss, grads = jax.value_and_grad(self._forward)(state, x, y)
        return loss, jax.tree_util.tree_map(self.optimizer, state, grads)

    def _forward(self, state, x, y):
        y_ = self._call_fn(state, x)
        return self.loss_fn(y_, y)
    
    def accuracy(self, y_hat, y):
        y_hat = jnp.argmax(y_hat, axis=1)
        return jnp.sum(y == y_hat) / y_hat.shape[0]

    def predict(self, x):
        y_hat = self._call_fn(self.state, x)
        return y_hat
