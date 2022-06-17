from jax import numpy as jnp


def mse():
    def _call(state, x, y, net):
        _, y_hat = net(state, x)
        return jnp.mean((y_hat - y)**2)

    return _call
