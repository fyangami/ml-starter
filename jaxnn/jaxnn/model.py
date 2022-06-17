from . import nn
import jax


class Model:
    def __init__(self, layers) -> None:
        self._init_fun, self._call_fn = nn.net(layers=layers)

    def fit(self, x, y, optimizer, loss_fn, epoch=1, *args, **kwargs):
        self.state = self._init_fun(x.shape[1:], *args, **kwargs)
        self.losses = []
        for i in range(epoch):
            loss, self.state = self.update(state=self.state,
                                           x=x,
                                           y=y,
                                           walk=jax.value_and_grad(
                                               loss_fn(self.state, x, y,
                                                       self._call_fn)),
                                           optimizer=optimizer)
            self.losses.append(loss)

    @staticmethod
    @jax.jit
    def update(state, x, y, walk, optimizer):
        loss, grads = walk(state, x, y)
        state = jax.tree_util.tree_map(optimizer, state, grads)
        return loss, state
