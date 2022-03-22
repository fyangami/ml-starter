from turtle import shape
from numpy import dtype
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class LeNet:

    def __init__(self, output, input_shape: tuple) -> None:
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
        self.weights = [self.kernel_1, self.kernel_1_bias, self.kernel_2, self.kernel_2_bias, self.dense_1, self.dense_1_bias, self.dense_2, self.dense_2_bias, self.dense_3, self.dense_3_bias]
    
    def start_timer(self):
        from time import time
        self.timer_begin = time() 
    
    def timer(self):
        from time import time
        interval = time() - self.timer_begin
        return interval
    
    def fit(self, train_iter: tf.data.Dataset, epochs=50, lr=1e-3):
        self.losses = []
        self.accuracies = []
        self.valid_accuracies = []
        self.epochs = epochs
        self.steps = len(train_iter)
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        for epoch in range(epochs):
            self.start_timer()
            for valid_x, valid_y in train_iter.take(1):
                pass
            for step, (x, y) in enumerate(train_iter):
                with tf.GradientTape() as gt:
                    y_ = self.__forward(x)
                    # loss
                    l_ = loss_fn(y, y_)
                grads = gt.gradient(l_, self.weights)
                optimizer.apply_gradients(zip(grads, self.weights))
                if step % 10 == 0:
                    self.losses.append(tf.reduce_mean(l_))
                    self.accuracies.append(self.accuracy(y, y_))
                    valid_y_ = self.__forward(valid_x)
                    self.valid_accuracies.append(self.accuracy(valid_y, valid_y_))
                    self.__print_fit_log(epoch + 1, step, self.losses[-1], self.accuracies[-1], self.valid_accuracies[-1])
            print()
            # print('epoch[%d\\%d loss: %.6f acc: %.6f' % (epoch + 1, step, epochs, self.losses[-1], self.accuracies[-1]))

    def __forward(self, x):
        y_ = self.net(x)
        y_ = tf.nn.softmax(y_)
        return y_
    
    def predict(self, x):
        y_ = self.__forward(x)
        return y_

    @staticmethod
    def accuracy(y, y_):
        y_ = tf.argmax(y_, axis=1, output_type=y.dtype)
        acc = y == y_
        return tf.reduce_sum(tf.cast(acc, tf.dtypes.int32)) / acc.shape[0]
    
    def __print_fit_log(self, epoch, step, loss, acc, valid_acc):
        procent = int(step / self.steps * 20)
        progress = '=' * (procent - 1) + '>' + ' ' * (20 - procent)
        print('\repoch[%2d\\%d speed\\%2.2fs (%s loss: %.6f acc: %.3f valid_acc: %.3f' % (epoch, self.epochs, self.timer(), progress, loss, acc, valid_acc), end='')


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


BATCH_SIZE = 50
EPOCHS = 6
ETA = 1e-3
if __name__ == '__main__':
    from mnist import train_images, train_labels, test_images, test_labels

    train_data = train_images()
    train_labels = train_labels()

    train_data = tf.constant(train_data / 255., dtype=tf.dtypes.float32)
    train_labels = tf.constant(train_labels,dtype=tf.dtypes.int32)
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    train_iter = tf.data.Dataset.zip((train_data, train_labels))
    train_iter = train_iter.shuffle(buffer_size=len(train_data), reshuffle_each_iteration=True).batch(batch_size = BATCH_SIZE, drop_remainder=True)
    net = LeNet(10, input_shape=(28, 28))
    net.fit(train_iter, epochs=EPOCHS, lr=ETA)

    test_data = test_images()
    test_labels = test_labels()
    test_data = tf.constant(test_data / 255., dtype=tf.dtypes.float32)
    test_labels = tf.constant(test_labels, dtype=tf.dtypes.int32)

    predicted = net.predict(test_data)
    print("terminaled[ test acc: %.3f" % net.accuracy(test_labels, predicted))
    
