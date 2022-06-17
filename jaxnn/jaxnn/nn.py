import jax
from jax import numpy as jnp
from .utils import random_key


def net(layers):
    def _init(n_in, *args, **kwargs):
        state = []
        for init, _ in layers:
            n_in, _state = init(n_in, *args, **kwargs)
            state.append(_state)
        return state

    def _call(state, x):
        for (_, call), _state in zip(layers, state):
            _, x = call(_state, x)
        return state, x
    
    return _init, _call


def relu():
    def _init(n_in, *args, **kwargs):
        return n_in, dict()

    def _call(state, x):
        return state, jnp.maximum(x, 0)

    return _init, _call


def flatten():
    def _init(n_in, *args, **kwargs):
        return n_in, dict()

    def _call(state, x):
        return state, jnp.reshape(x, (x.shape[0], sum(x.shape[1:])))


def dense(n_out):
    def _init(n_in, rng=random_key()):
        if isinstance(n_in, tuple):
            n_in = sum(n_in)
        w_key, b_key = jax.random.split(rng)
        return n_out, dict(w=jax.random.normal(w_key, (n_in, n_out)),
                    b=jax.random.normal(b_key, (n_out, )))

    def _call(state, x):
        return state, x @ state['w'] + state['b']

    return _init, _call
