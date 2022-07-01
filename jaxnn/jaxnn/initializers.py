import numpy as np
import jax

def glorot(in_axes=-2, out_axes=-1, type='normal', scaler=1e-2):
    
    def _call(rng, shape):
        if len(shape) == 1:
            fan_in, fan_out = shape[0], shape[0]
        else:  
            fan_in, fan_out = shape[in_axes], shape[out_axes]
        respective_field_size = 1
        if len(shape) > 2:
            respective_field_size = np.prod(shape) / fan_in / fan_out
        fan_in, fan_out = fan_in * respective_field_size,  fan_out * respective_field_size
        init_fn = getattr(jax.random, type)
        w = init_fn(rng, shape)
        return w * np.sqrt(2 / (fan_in + fan_out) * scaler)

    return _call


def normal(scaler=1e-2):
    
    def _call(rng, shape):
        return jax.random.normal(rng, shape) * scaler


def uniform(scaler=1e-2):
    
    def _call(rng, shape):
        return jax.random.uniform(rng, shape) * scaler
