import tensorflow as tf


class NN:
    def __init__(self) -> None:
        self.weights = []

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

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        for epoch in range(epochs):
            self.start_timer()
            for valid_x, valid_y in train_iter.take(1):
                pass
            for step, (x, y) in enumerate(train_iter):
                with tf.GradientTape() as gt:
                    y_ = self.__forward(x)
                    l_ = loss_fn(y, y_)
                grads = gt.gradient(l_, self.weights)
                optimizer.apply_gradients(zip(grads, self.weights))
                if step % 10 == 0:
                    self.losses.append(tf.reduce_mean(l_))
                    self.accuracies.append(self.accuracy(y, y_))
                    valid_y_ = self.__forward(valid_x)
                    self.valid_accuracies.append(
                        self.accuracy(valid_y, valid_y_))
                    self.__print_fit_log(epoch + 1, step, self.losses[-1],
                                         self.accuracies[-1],
                                         self.valid_accuracies[-1])
            print()
            # print('epoch[%d\\%d loss: %.6f acc: %.6f' % (epoch + 1, step, epochs, self.losses[-1], self.accuracies[-1]))

    def __forward(self, x):
        y_ = self.net(x)
        y_ = tf.nn.softmax(y_)
        return y_

    def predict(self, x):
        y_ = self.__forward(x)
        return tf.argmax(y_, axis=1)

    @staticmethod
    def accuracy(y, y_):
        y_ = tf.argmax(y_, axis=1, output_type=y.dtype)
        acc = y == y_
        return tf.reduce_sum(tf.cast(acc, tf.dtypes.int32)) / acc.shape[0]

    def __print_fit_log(self, epoch, step, loss, acc, valid_acc):
        procent = int(step / self.steps * 20)
        progress = '=' * (procent - 1) + '>' + ' ' * (20 - procent)
        print(
            '\repoch[%2d\\%d speed\\%2.2fs (%s loss: %.6f acc: %.3f valid_acc: %.3f'
            %
            (epoch, self.epochs, self.timer(), progress, loss, acc, valid_acc),
            end='')
