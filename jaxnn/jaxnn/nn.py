from functools import partial
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


def dense(n_out):

    def _init(n_in, rng):
        n_in = _flatten_dim(n_in)
        w_key, b_key = jax.random.split(rng)
        return n_out, dict(w=_normal(w_key, (n_in, n_out)),
                           b=_normal(b_key, (n_out, )))

    def _call(state, x, **kwargs):
        return state, x @ state['w'] + state['b']

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


def conv2d(n_filter: int, kernel_size, strides=(1, 1), padding='SAME'):
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
        state['w'] = _normal(w_rng,
                             shape=(*_kernel_size, kernel_input, n_filter))
        state['b'] = _normal(b_rng, shape=(n_filter, ))
        n_out = (out_h, out_w, n_filter)
        return n_out, state

    def _call(state, x, **kwargs):
        if len(x.shape) == 3:
            x = x[:, :, :, None]
        if len(x.shape) == 2:
            x = x[None, :, :, None]
        kernel = state['w']
        return state, jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=_strides,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + state['b']

    return _init, _call


def conv2d_old(n_filter: int, kernel_size, strides=(1, 1), padding='SAME'):
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
        state['k'] = jax.random.normal(rng, shape=(*_kernel_size, n_filter))
        n_out = (out_h, out_w, n_filter)
        return n_out, state

    def _call(state, x):
        if len(x.shape) == 3:
            x = x[:, :, :, None]
        if len(x.shape) == 2:
            x = x[None, :, :, None]
        assert len(x.shape) == 4
        return state, _conv2d(x=x,
                              kernel=state['k'],
                              strides=_strides,
                              padding=padding)

    return _init, _call


def _conv2d(x, kernel, strides, padding):

    def _conv_single(x, kernel):
        # x.shape=w, h
        sh, sw = strides
        h, w = x.shape
        kh, kw = kernel.shape
        if padding == 'SAME':
            hori = ((h * sh) - (h - kh + 1))
            vert = ((w * sw) - (w - kw + 1))
            left_padding = int(vert / 2)
            right_padding = vert - left_padding
            top_padding = int(hori / 2)
            bottom_padding = hori - top_padding
            x = jnp.concatenate([
                jnp.zeros((x.shape[0], left_padding)), x,
                jnp.zeros((x.shape[0], right_padding))
            ],
                                axis=1)
            x = jnp.concatenate([
                jnp.zeros((top_padding, x.shape[1])), x,
                jnp.zeros((bottom_padding, x.shape[1]))
            ],
                                axis=0)
            h, w = x.shape
        output = []
        for i in range(0, h - kh + 1, sh):
            row = []
            for j in range(0, w - kw + 1, sw):
                row.append(jnp.sum(x[i:i + kh, j:j + kw] * kernel))
            output.append(row)
        return jnp.array(output)

    def _conv_channels(x, kernel):
        # x.shape = h,w,c   kernel.shape=h,w
        convd = jax.vmap(_conv_single, in_axes=(2, None), out_axes=-1)(x,
                                                                       kernel)
        # convd.shape=h,w,c
        return jnp.sum(convd, axis=-1)

    def _conv_filters(x, kernel):
        # x.shape = h,w,c kernel.shape=h,w,f
        # convd.shape=h,w,c
        convd = jax.vmap(_conv_channels, in_axes=(None, 2),
                         out_axes=-1)(x, kernel)
        return convd

    convd = jax.vmap(_conv_filters, in_axes=(0, None))(x, kernel)
    return convd


def maxpool():

    def _init(n_in, rng):
        pass

    def _call(state, x, **kwargs):
        pass

    return _init, _call


def avgpool():

    def _init(n_in, rng):
        pass

    def _call(state, x, **kwargs):
        pass

    return _init, _call
