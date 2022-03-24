

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append("../../")
    from utils.preprocess import std
    data = pd.read_csv("../kaggle/song-popularity/song_data.csv").to_numpy()
    data_1 = std(data[:, 1:])
    from sklearn.preprocessing import StandardScaler
    data_2 = StandardScaler().fit_transform(data[:, 1:])
    print(data_1[0, :])
    print(data_2[0, :])
    print(data_1.max(axis=0))
    print(data_2.max(axis=0))
