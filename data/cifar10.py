from tensorflow.keras.datasets import cifar10

import tensorflow as tf


def __preprocess_x(x, resize=None):
    if resize:
        x = tf.image.resize(x, resize)
    return x / 255.0


def load_data(batch_size=50, resize=None):
    (x, y), (tx, ty) = cifar10.load_data()
    x = tf.constant(x, dtype='float32')
    y = tf.constant(y, dtype='float32')
    tx = tf.constant(tx, dtype='float32')
    ty = tf.constant(ty, dtype='float32')
    x = tf.data.Dataset.from_tensor_slices(x).map(
        lambda x: __preprocess_x(x, resize)
    )
    y = tf.data.Dataset.from_tensor_slices(y)
    tx = tf.data.Dataset.from_tensor_slices(tx).map(
        lambda x: __preprocess_x(x, resize)
    )
    ty = tf.data.Dataset.from_tensor_slices(ty)
    train_iter = (
        tf.data.Dataset.zip((x, y))
        .shuffle(buffer_size=len(x))
        .batch(batch_size=batch_size)
    )
    test_iter = tf.data.Dataset.zip((tx, ty))
    return train_iter, test_iter
