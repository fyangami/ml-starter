from numpy import ndarray
import numpy as np

class Liner:


    def __init__(self, eta=.00001) -> None:
        self.eta = eta
    
    def fit(self, X: ndarray, y: ndarray) -> None:
        X = np.c_[np.ones(len(X)), X]
        assert y.shape[1] == 1
        self.W = np.random.randn(1, X.shape[1])
        for i in range(100):
            y_hat = self.predict(X)
            gradients = self.gradients(X, y_hat, y)
            assert gradients.shape == (1, X.shape[1])
            self.W = self.W - self.eta * gradients
            loss = self.loss(y_hat, y)
            print("loss: %.6f" % loss)
            
    
    def predict(self, X) -> ndarray:
        # 10000 x 15   1 x 15
        return X.dot(self.W.T)

    def gradients(self, X, y_hat, y) -> ndarray:
        # 10000 x 15  10000 x 1 10000 x 15
        return (y_hat - y).T.dot(X)
    
    def loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2 / 2)


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../song-popularity/song_data.csv").to_numpy()
    song_names = data[:, 0]
    song_label = data[:, 1].reshape(-1, 1)
    song_data = data[:, 1:]
    liner = Liner()
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    song_data = std.fit_transform(song_data)
    liner.fit(song_data, song_label)
    
