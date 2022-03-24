import tensorflow as tf

from .nn import NN

class LeNet(NN):
    
    def __init__(self, output, input_shape: tuple) -> None:
        super().__init__()
        if len(input_shape) == 2:
            input_shape = (*input_shape, 1)
        assert len(input_shape) == 3
        initializer = tf.initializers.RandomUniform(minval=-.05, maxval=.05)
        self.kernel_1 = tf.Variable(initial_value=initializer(shape=(5, 5, input_shape[-1], 6)))
        self.kernel_1_bias = tf.Variable(tf.zeros(shape=(6,)))
        self.kernel_2 = tf.Variable(initial_value=initializer(shape=(5, 5, 6, 16)))
        self.kernel_2_bias = tf.Variable(tf.zeros(shape=(16,))) # 28 / 2 = 14
        from math import ceil
        flatten_in = ceil(input_shape[0] / 2 / 2) * ceil(input_shape[1] / 2 / 2) * 16
        self.dense_1 = tf.Variable(initial_value=initializer(shape=(flatten_in, 120)))
        self.dense_1_bias = tf.Variable(tf.zeros(shape=(1,)));
        self.dense_2 = tf.Variable(initial_value=initializer(shape=(120, 84)))
        self.dense_2_bias = tf.Variable(tf.zeros(shape=(1,)));
        self.dense_3 = tf.Variable(initial_value=initializer(shape=(84, output)))
        self.dense_3_bias = tf.Variable(tf.zeros(shape=(1,)));
        self.weights += (
            self.kernel_1, 
            self.kernel_1_bias, 
            self.kernel_2, 
            self.kernel_2_bias, 
            self.dense_1, 
            self.dense_1_bias, 
            self.dense_2, 
            self.dense_2_bias, 
            self.dense_3, 
            self.dense_3_bias
        )
 
 
    def net(self, x: tf.Tensor):
        if len(x.shape) == 3:
            x = tf.expand_dims(x, -1)
        assert len(x.shape) == 4
        # filter 6  kernel_size 5 activation sigmoid
        x = tf.nn.conv2d(x, self.kernel_1, strides=[1], padding='SAME') + self.kernel_1_bias
        x = tf.nn.sigmoid(x)
        # x.shape = (batch, h, w, 6)
        # avg pool 2 2
        x = tf.nn.avg_pool2d(x, ksize=[2], strides=[2], padding='SAME') 
        # x.shape = (batch, h, w, 6)
        # filter 16 kernel_size 5 activation sigmoid
        x = tf.nn.conv2d(x, self.kernel_2, strides=[1], padding='SAME') + self.kernel_2_bias
        x = tf.nn.sigmoid(x)
        # x.shape = (batch, h, w, 16)
        # avg pool 2 2
        x = tf.nn.avg_pool2d(x, ksize=[2], strides=[2], padding='SAME')
        # x.shape = (batch, h, w, 16)
        # flatten
        x = tf.reshape(x, shape=(x.shape[0], -1))

        # dense 120
        x = tf.matmul(x, self.dense_1) + self.dense_1_bias
        x = tf.nn.sigmoid(x)
        # dense 84
        x = tf.matmul(x, self.dense_2) + self.dense_2_bias
        x = tf.nn.sigmoid(x)
        # dense output
        x = tf.matmul(x, self.dense_3) + self.dense_3_bias
        return x
