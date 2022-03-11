import numpy as np
import pandas as pd

def data_genterator(x, y, seed=1024, batch_size=20):
    n_ = len(x)
    indicates = list(range(n_))
    np.random.seed(seed)
    np.random.shuffle(indicates)
    for batch in range(0, n_, batch_size):
        batcher = indicates[batch: min(batch + batch_size, n_)]
        yield x[batcher, :], y[batcher]

def one_hot(df: pd.DataFrame, cols, drop=True):
    for col in cols:
        unis = np.unique(df[col])
        for uni in unis:
            df[col + '_' + str(uni)] = (df[col] == uni).astype('uint8')
        if drop:
            df.drop(col, axis=1, inplace=True)
        
    
