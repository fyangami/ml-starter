from turtle import shape
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class LeNet:

    def __init__(self, input_shape: tuple) -> None:
        if len(input_shape) == 2:
            input_shape = (*input_shape, 1)
        assert len(input_shape) == 3
        initializer = tf.initializers.RandomUniform(minval=-.05, maxval=.05)
        self.kernel_1 = tf.Variable(initial_value=initializer(shape=(5, 5, input_shape[-1], 6)))
        self.kernel_1_bias = tf.Variable(tf.zeros(shape=(6,)))
        self.kernel_2 = tf.Variable(initial_value=initializer(shape=(5, 5, 6, 16)))
        self.kernel_2_bias = tf.Variable(tf.zeros(shape=(16,)))
        self.dense_1 = tf.Variable(initial_value=initializer(shape=(input_shape[0] * input_shape[1] * 16, 120)))
        self.dense_1_bias = tf.Variable(0);
        self.dense_2 = tf.Variable(initial_value=initializer(shape=(120, 84)))
        self.dense_2_bias = tf.Variable(0);
        self.dense_3 = tf.Variable(initial_value=initializer(shape=(84, 2)))
        self.dense_3_bias = tf.Variable(0);
        self.weights = [self.kernel_1, self.kernel_1_bias, self.kernel_2, self.kernel_2_bias, self.dense_1, self.dense_1_bias, self.dense_2, self.dense_2_bias, self.dense_3, self.dense_3_bias]
    
    def fit(self, train_iter: tf.data.Dataset, epochs=50, lr=1e-3):
        self.losses = []
        self.accuracies = []
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        for epoch in range(epochs):
            for x, y in train_iter:
                if len(x.shape) == 3:
                    x = tf.expand_dims(x, -1)
                assert len(x.shape) == 4               
                with tf.GradientTape() as gt:
                    y_ = self.net(x)
                    # softmax
                    y_ = tf.nn.softmax(y_)
                    # loss
                    l_ = loss_fn(y, y_)
                grads = gt.gradient(l_, self.weights)
                optimizer.apply_gradients(zip(grads, self.weights))
            self.losses.append(tf.reduce_mean(l_))
            self.accuracies.append(self.accuracy(y, y_))
            print('epoch[%d\\%d loss: %.6f' % (epoch + 1, epochs, self.losses[-1]))

    @staticmethod
    def accuracy(y, y_):
        y_ = tf.argmax(y_, axis=1, output_type=y.dtype)
        acc = y == y_
        return tf.reduce_sum(tf.cast(acc, tf.dtypes.int8)) / acc.shape[0]


def net(self, x: tf.Tensor):
    # filter 6  kernel_size 5 activation sigmoid
    x = tf.nn.conv2d(x, self.kernel_1, strides=[1], padding='SAME') + self.kernel_1_bias
    x = tf.nn.relu(x)
    # x.shape = (batch, h, w, 6)
    # avg pool 2 2
    x = tf.nn.avg_pool2d(x, ksize=[2], strides=[2], padding='SAME') 
    # x.shape = (batch, h, w, 6)
    # filter 16 kernel_size 5 activation sigmoid
    x = tf.nn.conv2d(x, self.kernel_2, strides=[1], padding='SAME') + self.kernel_2_bias
    x = tf.nn.relu(x)
    # x.shape = (batch, h, w, 16)
    # avg pool 2 2
    x = tf.nn.avg_pool2d(x, ksize=[2], strides=[2], padding='SAME')
    # x.shape = (batch, h, w, 16)
    # flatten
    x = tf.reshape(x, shape=(x.shape[0], -1))

    # dense 120
    x = tf.matmul(x, self.dense_1) + self.dense_1_bias
    x = tf.nn.relu(x)
    # dense 84
    x = tf.matmul(x, self.dense_2) + self.dense_2_bias
    x = tf.nn.relu(x)
    # dense output
    x = tf.matmul(x, self.dense_3) + self.dense_3_bias
    return x


if __name__ == '__main__':
    import tensorflow_datasets as tfds
    (train_iter, test_iter), info = tfds.load(
        'mnist', 
        split=['train', 'test'], 
        shuffle_files=True, 
        as_supervised=True, 
        with_info=True,
        batch_size=50
    )
    for x, y in train_iter:
        print(x)
        print(y)
        break

