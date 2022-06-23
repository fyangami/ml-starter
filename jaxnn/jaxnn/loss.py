from jax import numpy as jnp
import jax


def mean_squared_error():
    def _call(y_hat, y):
        return jnp.mean((y_hat - y)**2)

    return _call

def categorical_cross_entropy(from_logits=False):
    def _call(y_hat, y):
        y_hat = jnp.clip(y_hat, 1e-9, 1.)
        y = jax.nn.one_hot(y, y_hat.shape[-1])
        if from_logits:
            return -jnp.mean(
                jnp.sum(y * y_hat, axis=1))
        else:
            return -jnp.mean(
                jnp.sum(y * jnp.log(y_hat), axis=1))

    return _call
