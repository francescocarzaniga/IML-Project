import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_to_numerical(array):
    label_numerical = []
    for column in range(array.shape[1]):
        try:
            feature = np.asarray(array[:, column]).astype(float)
            label_numerical.append(feature)
        except ValueError:
            le = LabelEncoder()
            feature = le.fit_transform(array[:, column])
            label_numerical.append(feature)
    return np.stack(label_numerical, axis=1)