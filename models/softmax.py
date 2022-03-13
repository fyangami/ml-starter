import numpy as np
from utils.data import one_hot
from utils.data import data_genterator

class SoftmaxClassifier:
    
    def __init__(self, C=1.0, eta=1e-3, epochs=50, batch_size=50, penalty=False) -> None:
        self.C = C
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.losses = np.zeros(epochs)
        self.accuracies = np.zeros(epochs)
    
    def fit(self, x, y):
        self._fit_init(x, y)
        for epoch in range(self.epochs):
            self._fit_epoch(x, y)
            predicted = self.__predict(x)
            loss = self.__cross_entropy_loss(predicted ,y)
            self.losses[epoch] = loss
            acc = self.__accuracy(predicted, y)
            self.accuracies[epoch] = self.__accuracy(predicted , y)
            print('epoch[%d/%d loss: %.6f - acc: %.6f' % (epoch + 1, self.epochs, loss, acc))
    
    def _fit_epoch(self, x, y):
        for batch_x, batch_y in data_genterator(x, y):
            y_hat = self.__predict(batch_x)
            grad_w, grad_b = self.__gradients(y_hat, batch_y, batch_x)
            if self.penalty:
                grad_w = grad_w + self.C * self.w
            self.__update(grad_w, grad_b)
        
    def __update(self, grad_w, grad_b):
        self.w = self.w - self.eta * grad_w
        self.b = self.b - self.eta * grad_b

    def _fit_init(self, x, y):
        self.n_class = len(np.unique(y))
        self.n_features = x.shape[1]
        self.w = np.random.randn(self.n_class, self.n_features)
        self.b = np.random.randn(1, self.n_class)

    def __predict(self, x):
        logits = np.dot(x, self.w.T) + self.b
        exp = np.exp(logits)
        return  exp / np.sum(exp, axis=1, keepdims=True)

    def __accuracy(self, y_hat, y):
        y_hat = np.argmax(y_hat, axis=1, keepdims=True).astype(y.dtype)
        return (y_hat == y).sum() / len(y)

    def __cross_entropy_loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-9, 1.)
        y = one_hot(y, n_=self.n_class)
        loss = -np.mean(np.sum(y * np.log(y_hat), axis=1))
        if self.penalty:
            loss = loss + np.mean(self.w ** 2) * self.C
        return loss

    def __gradients(self, y_hat, y, x):
        y = one_hot(y, n_=self.n_class)
        return -((y - y_hat) * y).T.dot(x), -np.mean((y - y_hat) * y)
    
    def predict(self, x):
        predicted = self.__predict(x)
        return np.argmax(predicted, axis=1)
    
    def accuracy(self, x, y):
        predicted = self.__predict(x)
        return self.__accuracy(predicted, y)
