import numpy as np
import pandas as pd

def data_genterator(x, y, seed=None, batch_size=20):
    n_ = len(x)
    indicates = list(range(n_))
    if seed:
        np.random.seed(seed)
    np.random.shuffle(indicates)
    for batch in range(0, n_, batch_size):
        batcher = indicates[batch: min(batch + batch_size, n_)]
        yield x[batcher, :], y[batcher]

def one_hot_df(df: pd.DataFrame, cols, drop=True):
    for col in cols:
        unis = np.unique(df[col])
        for uni in unis:
            df[col + '_' + str(uni)] = (df[col] == uni).astype('uint8')
        if drop:
            df.drop(col, axis=1, inplace=True)


def one_hot(y, n_):
    assert y.shape[1] == 1
    yy = None
    for uni in range(n_):
        if yy is None:
            yy = y == uni
        else:
            yy = np.c_[yy, y == uni]
    return yy.astype('uint8')
