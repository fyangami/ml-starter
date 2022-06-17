from . import nn
import jax


class Model:
    def __init__(self, layers) -> None:
        self._init_fn, self._call_fn = nn.net(layers=layers)

    def fit(self, x, y, optimizer, loss_fn, epoch=1, *args, **kwargs):
        self.state = self._init_fn(x.shape[1:], *args, **kwargs)
        self.losses = []
        # self.__update_fn = jax.jit(self.update, static_argnames=['walk', 'optimizer', 'net'])
        self.__update_fn = self.update
        # self.__update_fn = jax.jit(self.update, static_argnames=['walk', 'optimizer', 'net'])
        for i in range(epoch):
            loss, self.state = self.__update_fn(state=self.state,
                                           x=x,
                                           y=y,
                                           walk=jax.value_and_grad(
                                               loss_fn, argnums=[0, 1, 2]),
                                           optimizer=optimizer,
                                           net=self._call_fn)
            self.losses.append(loss)

    @staticmethod
    def update(state, x, y, walk, optimizer, net):
        loss, grads = walk(state, x, y, net)
        state = jax.tree_util.tree_map(optimizer, state, grads)
        return loss, state
