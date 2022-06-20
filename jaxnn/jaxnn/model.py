from functools import partial

from jaxnn.utils import random_key
from . import nn
import jax


class Model:
    def __init__(self, layers) -> None:
        self._init_fn, self._call_fn = nn.net(layers=layers)
        self.state = None


    def fit(self, data_loader, optimizer, loss_fn, epochs=1, rng=random_key(), *args, **kwargs):
        self.losses = []
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        for epoch in range(epochs):
            train_iter, test_data = next(data_loader)
            for (x, y) in train_iter:
                loss = self.step(x, y, rng)
                self.losses.append(loss)
            print(f'epoch: [{epoch + 1}\\{epochs} -- loss: {loss:<.3f}')

    def step(self, x, y, rng):
        if self.state is None:
            self.state = self._init_fn(x.shape[1:], rng)
        loss, self.state = self._update(state=self.state,
                                            x=x,
                                            y=y)
        return loss

    @partial(jax.jit, static_argnums=0)
    def _update(self, state, x, y):
        loss, grads = jax.value_and_grad(self._forward)(state, x, y)
        return loss, jax.tree_util.tree_map(self.optimizer, state, grads)

    def _forward(self, state, x, y):
        _, y_ = self._call_fn(state, x)
        return self.loss_fn(y_, y)

    def predict(self, x):
        _, y_hat = self._call_fn(self.state, x)
        return y_hat
