from mnist import train_images, train_labels, test_images, test_labels
from jax import numpy as jnp
import jax
from jaxnn import utils


def mnist_dataloader(rng=utils.random_key(), batch_size=50, repeat=True, valid_ratio=.2, map_fn=None, *args, **kwargs):
    x = train_images()
    if map_fn is not None:
        x = map_fn(x)
    y = train_labels()
    valid_len = int(valid_ratio * x.shape[0])
    while True:
        shuffled_x = jax.random.permutation(rng, x, axis=0)
        shuffled_y = jax.random.permutation(rng, y, axis=0)
        valid_x = shuffled_x[:valid_len]
        valid_y = shuffled_y[:valid_len]
        yield _split(shuffled_x[valid_len:], shuffled_y[valid_len:], rng, batch_size), (valid_x, valid_y)
        if not repeat:
            break
        (rng, _) = jax.random.split(rng)

def mnist_testdata():
    return test_images(), test_labels()


def _split(x, y, rng, batch_size):
    all = len(x)
    n_batch = all // batch_size
    for i in range(n_batch):
        left = batch_size * i
        right = left + batch_size
        if right <= all:
            yield x[left:right], y[left:right]
    rng, _ = jax.random.split(rng)


if __name__ == '__main__':
    data_loader = mnist_dataloader()
    train_iter, _ = next(data_loader)
    x, y = next(train_iter)
    print(x, y)
