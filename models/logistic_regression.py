from numpy import ndarray
import numpy as np


class LogisticClassifier:

    def __init__(self, eta=1e-3, epochs=50, batch_size=20, seed=1024) -> None:
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.__seed = seed
    
    def logic(self, x: ndarray) -> ndarray:
        # 20 x 20  20 x 1
        print(x.shape)
        print(self.w.shape)
        liner = x.dot(self.w) + self.b
        return 1 / (1 + np.exp(-liner))
    
    def loss(self, y_hat, y) -> ndarray:
        return (y_hat - y) ** 2 / 2
    
    def gradients(self, x, y_hat, y):
        return (y_hat - y).T.dot(x), (y_hat - y)
    
    def fit(self, x, y, tx, ty) -> None:
        np.random.seed(self.next_seed())
        self.w = np.random.randn(x.shape[1], 1)
        self.next_seed()
        self.b = np.random.randn(1)
        self.losses = np.zeros(self.epochs)
        from utils.data import data_genterator
        for epoch in range(self.epochs):
            for x, y in data_genterator(x, y, self.next_seed(), self.batch_size):
                y_hat = self.logic(x)
                dw, db = self.gradients(x, y_hat, y)
                self.w = self.w - (self.eta * dw / len(x)).mean(axis=0)
                self.b = self.b - (self.eta * db / len(x)).mean(axis=0)
            l = self.loss(self.logic(tx), ty)
            self.losses[epoch] = np.mean(l)
            print(f'epoch: {epoch + 1} --- loss: {self.losses[epoch]}')
    
    def next_seed(self):
        seed = self.__seed
        self.__seed += 1
        return seed
