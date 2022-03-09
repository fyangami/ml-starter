from numpy import ndarray
import numpy as np
from utils.preprocess import shuffle

class Liner:

    def __init__(self, eta=1e-7, epochs=50, batch_size=20) -> None:
        self.eta = eta
        self.epochs = epochs
        self.batch_size = 20

    def fit(self, X: ndarray, y: ndarray, tX: ndarray, ty: ndarray) -> None:
        X = np.c_[np.ones(len(X)), X]
        tX = np.c_[np.ones(len(tX)), tX]
        self.W = np.random.randn(1, X.shape[1])
        self.losses = np.zeros(self.epochs)
        self.losses_t = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            self._fit_epoch(X, y)
            self.losses[epoch] = self.loss(self.predict(X), y)
            self.losses_t[epoch] = self.loss(self.predict(tX), ty)
            print("epoch[%d] train loss: %.6f, test loss: %.6f" % (epoch, self.losses[epoch], self.losses_t[epoch]))
    
    def _fit_epoch(self, X, y):
        X, y = shuffle(X, y)
        iters = len(X) // self.batch_size
        for i in range(iters):
            l = i * self.batch_size
            r = l + self.batch_size
            if r > len(X):
                r = len(X) - l
            batch_X = X[l:r, :]
            batch_y = y[l:r, :]
            self._fit(batch_X, batch_y)
    
    def _fit(self, X, y):
        y_hat = self.predict(X)
        gradients = self.gradients(X, y_hat, y)
        self.W = self.W - self.eta * gradients

    def predict(self, X) -> ndarray:
        # x: 100 x 15  w: 1 x 15  w.T: 15 x 1
        return X.dot(self.W.T)

    def gradients(self, X, y_hat, y) -> ndarray:
        # y:100 x 1   x:100 x 15
        # y.T: 1 x 100
        return ((y_hat - y) * X).mean(axis=0)
        # return (y_hat - y).T.dot(X)
    
    def loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2 / 2)
