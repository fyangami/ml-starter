from mnist import train_images, train_labels, test_images, test_labels
from jax import numpy as jnp
import jax
from jaxnn import utils


def mnist(rng=utils.random_key(), batch_size=50, repeat=True):
    x = train_images() / 255.
    y = train_labels()
    tx = test_images() / 255.
    ty = test_labels()
    while True:
        rng_1, rng_2 = jax.random.split(rng)
        yield _split(x, y, rng_1, batch_size), (tx, ty)
        if not repeat:
            break
        rng = rng_1


def _split(x, y, rng, batch_size):
    shuffled_x = jax.random.permutation(rng, x, axis=0)
    shuffled_y = jax.random.permutation(rng, y, axis=0)
    all = len(x)
    n_batch = all // batch_size
    for i in range(n_batch):
        left = batch_size * i
        right = left + batch_size
        if right <= all:
            yield shuffled_x[left:right], shuffled_y[left:right]
    rng, _ = jax.random.split(rng)


if __name__ == '__main__':
    data_loader = mnist()
    train_iter, _ = next(data_loader)
    x, y = next(train_iter)
    print(x, y)
