import jax
from jax import numpy as jnp
from .utils import random_key


def _flatten_dim(dim):
    if not isinstance(dim, int):
        flatten_dim = 1
        for dim_ in dim:
            flatten_dim *= dim_
        dim = flatten_dim
    return dim


def net(layers):
    def _init(n_in, rng):
        state = []
        for init, _ in layers:
            print(f"n_in: {n_in}")
            n_in, _state = init(n_in, rng)
            state.append(_state)
        return state

    def _call(state, x):
        for (_, call), _state in zip(layers, state):
            _, x = call(_state, x)
        return state, x

    return _init, _call


def relu():
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x):
        return state, jnp.maximum(x, 0)

    return _init, _call


def flatten():
    def _init(n_in, rng):
        return _flatten_dim(n_in), ()

    def _call(state, x):
        return state, jnp.reshape(x, (x.shape[0], _flatten_dim(x.shape[1:])))

    return _init, _call


def dense(n_out):
    def _init(n_in, rng):
        n_in = _flatten_dim(n_in)
        w_key, b_key = jax.random.split(rng)
        return n_out, dict(w=jax.random.normal(w_key, (n_in, n_out)),
                           b=jax.random.normal(b_key, (n_out, )))

    def _call(state, x):
        return state, x @ state['w'] + state['b']

    return _init, _call


def sortmax():
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x):
        exp = jnp.exp(x)
        return state, exp / jnp.sum(exp, axis=1, keepdims=True)

    return _init, _call
