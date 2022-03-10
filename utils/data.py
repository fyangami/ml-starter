import numpy as np

def data_genterator(x, y, seed=1024, batch_size=20):
    n_ = len(x)
    indicates = list(range(n_))
    np.random.seed(seed)
    np.random.shuffle(indicates)
    for batch in range(0, n_, batch_size):
        batcher = indicates[batch: min(batch + batch_size, n_)]
        yield x[batcher, :], y[batcher]
    