import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


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


def impute_whole(array):
    dataset = []
    for column in range(array.shape[1]):
        try:
            imp = SimpleImputer(strategy='median')
            feature = np.asarray(array[:, column]).astype(float)
            feature = imp.fit_transform(feature)
            dataset.append(feature.ravel())
        except ValueError:
            imp = SimpleImputer(strategy='most_frequent')
            feature = np.asarray(array[:, column]).astype(object).reshape(-1, 1)
            feature = imp.fit_transform(feature)
            dataset.append(feature.ravel())
    return np.stack(dataset, axis=1)
