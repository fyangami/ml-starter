import jax
from jax import numpy as jnp
from . import utils


def _normal(rng, shape, stddev=1e-3):
    return jax.random.normal(rng, shape) * stddev


def _flatten_dim(dim):
    if not isinstance(dim, int):
        flatten_dim = 1
        for dim_ in dim:
            flatten_dim *= dim_
        dim = flatten_dim
    return dim


_expand_2d = lambda w: (w, w) if isinstance(w, int) else w


def net(layers):
    def _init(n_in, rng):
        state = []
        for init, _ in layers:
            n_in, _state = init(n_in, rng)
            rng, _ = jax.random.split(rng)
            state.append(_state)
        return state

    def _call(state, x, rng=None, training=True):
        rng = rng or utils.random_key()
        for (_, call), _state in zip(layers, state):
            rng, _ = jax.random.split(rng)
            _, x = call(_state, x, rng=rng, training=training)
        return state, x

    return _init, _call


def relu():
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x, **kwargs):
        return state, jnp.maximum(x, 0)

    return _init, _call


def flatten():
    def _init(n_in, rng):
        return _flatten_dim(n_in), ()

    def _call(state, x, **kwargs):
        return state, jnp.reshape(x, (x.shape[0], _flatten_dim(x.shape[1:])))

    return _init, _call


def dense(n_out, initializer=None, with_bias=True):
    def _init(n_in, rng):
        n_in = _flatten_dim(n_in)
        w_key, b_key = jax.random.split(rng)
        return n_out, dict(w=initializer(w_key, (n_in, n_out))
                           if initializer else _normal(w_key, (n_in, n_out)),
                           b=initializer(b_key, (n_out, )) if initializer else
                           _normal(b_key, (n_out, )) if with_bias else None)

    def _call(state, x, **kwargs):
        x = x @ state['w']
        if with_bias:
            x = x + state['b']
        return state, x

    return _init, _call


def softmax():
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x, **kwargs):
        x_max = jnp.max(x, axis=1, keepdims=True)
        exp = jnp.exp(x - jax.lax.stop_gradient(x_max))
        return state, exp / jnp.sum(exp, axis=1, keepdims=True)

    return _init, _call


def dropout(threshold=.5):
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x, **kwargs):
        training = kwargs.get('training', False)
        if not training:
            return state, x
        rng = kwargs['rng']
        filter = jax.random.bernoulli(rng, shape=x.shape)
        return state, jnp.where(filter, x / threshold, 0)

    return _init, _call


def log_softmax():
    def _init(n_in, rng):
        return n_in, ()

    def _call(state, x, **kwargs):
        x_max = jnp.max(x, axis=1, keepdims=True)
        shifted = x - jax.lax.stop_gradient(x_max)
        shifted_logsumexp = jnp.log(
            jnp.sum(jnp.exp(shifted), axis=1, keepdims=True))
        return state, shifted - shifted_logsumexp

    return _init, _call


def conv2d(n_filter: int,
           kernel_size,
           strides=(1, 1),
           padding='SAME',
           initializer=None,
           with_bias=True):
    if isinstance(strides, int):
        _strides = (strides, strides)
    else:
        _strides = strides
    if isinstance(kernel_size, int):
        _kernel_size = [kernel_size, kernel_size]
    else:
        _kernel_size = kernel_size
    assert len(_kernel_size) == 2

    def _init(n_in, rng):
        out_h, out_w = n_in[:2]
        state = dict()
        kernel_input = 1 if len(n_in) < 3 else n_in[-1]
        w_rng, b_rng = jax.random.split(rng)
        w_shape = (*_kernel_size, kernel_input, n_filter)
        b_shape = (n_filter, )
        state['w'] = initializer(w_rng, w_shape) if initializer else _normal(
            w_rng, shape=w_shape)
        if with_bias:
            state['b'] = initializer(
                b_rng, b_shape) if initializer else _normal(b_rng,
                                                            shape=b_shape)
        n_out = (out_h, out_w, n_filter)
        return n_out, state

    def _call(state, x, **kwargs):
        if len(x.shape) == 3:
            x = x[:, :, :, None]
        if len(x.shape) == 2:
            x = x[None, :, :, None]
        kernel = state['w']
        x = jax.lax.conv_general_dilated(lhs=x,
                                         rhs=kernel,
                                         window_strides=_strides,
                                         padding=padding,
                                         dimension_numbers=('NHWC', 'HWIO',
                                                            'NHWC'))
        if with_bias:
            x = x + state['b']
        return state, x

    return _init, _call


def _reduce_window2d(window, strides, padding, init_val, op):
    _window = _expand_2d(window) + (1, )
    _strides = _expand_2d(strides) + (1, )

    def _init(n_in, rng):
        padding_vals = jax.lax.padtype_to_pads(n_in, _window, _strides,
                                               padding)
        n_out = jax.lax.reduce_window_shape_tuple(operand_shape=n_in,
                                                  window_dimensions=_window,
                                                  window_strides=_strides,
                                                  padding=padding_vals)
        return n_out, ()

    def _call(state, x, **kwargs):
        return state, jax.lax.reduce_window(operand=x,
                                            init_value=init_val,
                                            window_dimensions=(1, ) + _window,
                                            window_strides=(1, ) + _strides,
                                            padding=padding,
                                            computation=op)

    return _init, _call


def maxpool2d(window, strides, padding='VALID'):
    return _reduce_window2d(window,
                            strides,
                            padding,
                            init_val=-jnp.inf,
                            op=jax.lax.max)
