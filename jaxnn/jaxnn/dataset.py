import random
from urllib.request import urlretrieve
import zipfile
from mnist import train_images, train_labels, test_images, test_labels
from jax import numpy as jnp
import jax
from jaxnn import utils
import os
import re
from PIL import Image

DATA_SET_BASE_DIR = '/tmp/jaxnn_datasets/'

DATASETS = {
    'cat_and_dog': {
        'url':
        'https://storage.googleapis.com/kaggle-data-sets/23777/30378/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220623%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220623T031634Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=29c6467d49300db347b352e80eec5f53501c22a431e10d5a9d9e80d2f450e50471c3d4d2b5fc55c85d7d4643dd601df9c123b61eec2f0c962ccfd7a98acc2fb1ba741cea8a2f863004913cc7fbbf781d3deabe276a041479e5ed4936a9273483d926a6c4d433ea54de1be8d080da389975ab4f31e0f281430a55f7f49c30372faa7c343f60e7520d1eeba67aa3e05dd9fa39f279369c861b65eae0dcac6a4b71090e29b8315a69959ec01a2029f47a53f54a9242b787327d916785353934a0832daf1bef47996f9f3c13765d966f0ecdfed1767a7f62599b2fc0a000547e771f4c1c53b0bab4f63f25646cbc39cca442e2ae1f325697d04d6a8b4b73ef3a1725',
        'filename': 'cat-and-dog.zip'
    }
}


def _path_reveal(filename):
    if not os.path.exists(DATA_SET_BASE_DIR):
        os.makedirs(DATA_SET_BASE_DIR)
    return DATA_SET_BASE_DIR + filename


def mnist_dataloader(rng=utils.random_key(),
                     batch_size=50,
                     repeat=True,
                     valid_ratio=.2,
                     map_fn=None,
                     *args,
                     **kwargs):
    return _dataloader(x=train_images(),
                       y=train_labels(),
                       batch_size=batch_size,
                       rng=rng,
                       repeat=repeat,
                       valid_ratio=valid_ratio,
                       map_fn=map_fn)


def _dataloader(x, y, batch_size, repeat, valid_ratio, map_fn, rng):
    valid_len = int(valid_ratio * x.shape[0])
    while True:
        shuffled_x = jax.random.permutation(rng, x, axis=0)
        shuffled_y = jax.random.permutation(rng, y, axis=0)
        valid_x = shuffled_x[:valid_len]
        valid_y = shuffled_y[:valid_len]
        if map_fn is not None:
            shuffled_x = map_fn(shuffled_x)
            valid_x = map_fn(valid_x)
        yield _split(shuffled_x[valid_len:], shuffled_y[valid_len:], rng,
                     batch_size), (valid_x, valid_y)
        if not repeat:
            break
        (rng, _) = jax.random.split(rng)


def mnist_testdata():
    return test_images(), test_labels()


def _get_file(remote_url, filename):
    filename = _path_reveal(filename)
    if os.path.exists(filename):
        return open(filename, 'rb+')
    urlretrieve(url=remote_url, filename=filename)
    return open(filename, 'rb+')


def cat_and_dog_dataloader(size=(256, 256),
                rng=utils.random_key(),
                batch_size=50,
                repeat=True,
                valid_ratio=.2,
                map_fn=None):
    dataset = DATASETS['cat_and_dog']
    file_data = _get_file(remote_url=dataset['url'],
                          filename=dataset['filename'])
    all = zipfile.ZipFile(file_data)

    def _load_data(all: zipfile.ZipFile):
        dataset = [(re.compile(r'^train.*cat.*.jpg$'), []),
                   (re.compile(r'^train.*dog.*.jpg$'), []),
                   (re.compile(r'^test.*cat.*.jpg$'), []),
                   (re.compile(r'^test.*dog.*.jpg$'), [])]
        for fi in all.filelist:
            for (matcher, coll) in dataset:
                if matcher.match(fi.filename):
                    coll.append(fi)
                    # img = Image.open(all.open(fi))
                    # img = jnp.array(img)
                    # img = jax.image.resize(img, (*size, 3), method='nearest')
                    # coll.append(img)
        return (pair[1] for pair in dataset)

    train_cat, train_dog, test_cat, test_dog = _load_data(all)
    train = train_cat + train_dog
    x = jnp.arange(len(train_cat) + len(train_dog))

    # x = jnp.stack((train_cat, train_dog))
    y = jnp.concatenate((jnp.full(shape=(len(train_cat), ), fill_value=0),
                   jnp.full(shape=(len(train_dog), ), fill_value=1)))
    tx = jnp.arange(len(test_cat) + len(test_dog))
    # tx = jnp.stack((test_cat, test_dog))
    ty = jnp.concatenate(
        (jnp.full(shape=(len(test_cat), ),
                  fill_value=0), jnp.full(shape=(len(test_dog), ),
                                          fill_value=1)))

    del train_cat, train_dog, test_cat, test_dog
    def _lazy_map(lazy_x):
        xs = []
        for i in lazy_x:
            if isinstance(train[i], zipfile.ZipInfo):
                img = Image.open(all.open(train[i]))
                img = jnp.array(img)
                img = jax.image.resize(img, (*size, 3), method='nearest')
                train[i] = img
            xs.append(train[i])
        lazy_x = jnp.stack(xs)
        if map_fn is not None:
            lazy_x = map_fn(lazy_x)
        return lazy_x
    return _dataloader(x=x,
                       y=y,
                       batch_size=batch_size,
                       rng=rng,
                       repeat=repeat,
                       valid_ratio=valid_ratio,
                       map_fn=_lazy_map), (tx, ty)


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
    (train_iter, (tx, ty)) = cat_and_dog_dataloader()
    (x, y) = next(train_iter)
    print(x, y)
