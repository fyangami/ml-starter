from numpy import ndarray
import numpy as np
from utils.data import one_hot_y


class LogisticClassifier:

    def __init__(self, eta=1e-3, epochs=50, batch_size=20, seed=1024, verbose=True) -> None:
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.__seed = seed
        self.verbose = verbose
    
    def logic(self, x: ndarray) -> ndarray:
        liner = x.dot(self.w) + self.b
        return 1 / (1 + np.exp(-liner))
    
    def loss(self, y_hat, y) -> ndarray:
        return -((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))).mean()
    
    def gradients(self, x, y_hat, y):
        return (y_hat - y).T.dot(x), (y_hat - y)
    
    def fit(self, x, y, tx, ty) -> None:
        np.random.seed(self.next_seed())
        self.w = np.random.randn(x.shape[1], 1)
        self.b = np.random.randn(1)
        self.losses = np.zeros(self.epochs)
        from utils.data import data_genterator
        for epoch in range(self.epochs):
            for x, y in data_genterator(x, y, self.next_seed(), self.batch_size):
                y_hat = self.logic(x)
                dw, db = self.gradients(x, y_hat, y)
                self.w = self.w - (self.eta * dw / len(x)).mean(axis=0).reshape(-1, 1)
                self.b = self.b - (self.eta * db / len(x)).mean(axis=0).reshape(-1, 1)
            l = self.loss(self.logic(tx), ty)
            self.losses[epoch] = np.mean(l)
            if self.verbose:
                print(f'epoch: {epoch + 1} --- loss: {self.losses[epoch]}')
    
    def next_seed(self):
        seed = self.__seed
        self.__seed += 1
        return seed


class MultiLogisticClassifier:
    
    def __init__(self, eta=1e-3, epochs=50, batch_size=20, seed=1024, verbose=False) -> None:
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.__seed = seed
        self.verbose=verbose


    def fit(self, x, y, tx, ty):
        classes = np.unique(y)
        print(f'got classes: {classes}, ues one to many classifier')
        self.models = list()
        for i, n_class in enumerate(classes):
            n_model = LogisticClassifier(self.eta, self.epochs, self.batch_size, self.__seed, verbose=self.verbose)
            n_model.fit(x, (y == n_class).astype('uint8').reshape(-1, 1), tx, (ty == n_class).astype('uint8').reshape(-1, 1))
            self.models.append(n_model)
            print(f'iter[{i}, complete == {(i + 1) / len(classes) * 100}%]')

    def predict(self, x):
        predicted = np.zeros(len(x))
        for n_class, model in enumerate(self.models):
            n_predcit = np.round(model.logic(x))
            for idx, predict in enumerate(n_predcit):
                if predict == 1.:
                    predicted[idx] = n_class
            
        return predicted

    def score(self, x):
        predicted = None# np.zeros((x,len(self.model_map)))
        for model in self.models:
            n_predict = model.logic(x)
            if predicted is None:
                predicted = n_predict
            else:
                predicted = np.c_[predicted, n_predict]
        return predicted
    
    def losses(self, idx) -> ndarray:
        return self.models[idx].losses
