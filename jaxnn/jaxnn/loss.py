from jax import numpy as jnp


def mean_squared_error():
    def _call(y_hat, y):
        return jnp.mean((y_hat - y)**2)
    return _call
