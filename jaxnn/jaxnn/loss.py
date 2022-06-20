from jax import numpy as jnp
import jax


def mean_squared_error():
    def _call(y_hat, y):
        return jnp.mean((y_hat - y)**2)

    return _call


def categorical_cross_entropy():
    def _call(y_hat, y):
        one_hot_y = jax.nn.one_hot(y, jnp.max(y) + 1)
        return -jnp.mean(one_hot_y * jnp.log(y_hat))

    return _call


def _softmax(logits):
    exp = jnp.exp(logits)
    return exp / jnp.sum(exp, axis=1, keepdims=True)
