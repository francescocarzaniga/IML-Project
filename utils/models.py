import numpy as np
import pandas as pd
from utils.preprocessing import label_to_numerical


class DecisionTree(object):
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = []
        self._queue = []

    @staticmethod
    def __entropy(labels):
        if labels.size == 0:
            return 0
        unique, counts = np.unique(labels, return_counts=True)
        return np.sum([-counts[i]/np.sum(counts)*np.log2(counts[i]/np.sum(counts)) for i in range(len(unique))])

    def __gain(self, y, subsets):
        entropy_node = self.__entropy(y)
        total_length = np.sum([len(subset)] for subset in subsets)
        weights = [len(subset)/total_length for subset in subsets]
        entropy_child = np.sum([weights[i]*self.__entropy(y[subsets[i]]) for i in range(len(subsets))])
        return entropy_node-entropy_child

    def __split(self, X, y, features, max_features=None):
        if features.size > max_features:
            random_features = np.random.choice(features, max_features)
        else:
            random_features = features
        max_gain = 0.
        max_feature = -1
        threshold = None
        y_features = np.unique(y[features])
        if y_features.size == 1:
            self.tree.append(y_features[0])
            return
        for feature in random_features:
            feature_vector = X[:, feature]
            unique = np.unique(feature_vector)
            if type(feature_vector) is str:
                subsets = [np.where(feature_vector == u) for u in unique]
                gain = self.__gain(y, subsets)
            elif type(feature_vector) is float:
                threshold_gains = [self.__gain(y, [np.where(feature_vector <= u), np.where(feature_vector > u)])
                                   for u in unique]
                gain = np.max(threshold_gains)
                threshold = unique[np.argmax(threshold_gains)]
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
        if max_gain == 0.:
            self.tree[-1] = round(float(np.mean(y[features])))
        self.tree.append((max_feature, threshold))
        self._queue.append((max_features, threshold))
        return

    def fit(self, X, y, max_features=None):
        self.__split(X, y, np.arange(X.shape[1]), max_features=max_features)
        while len(self._queue) > 0:
            node_feat, node_thresh = self._queue.pop()
            if node_thresh is None:
                unique = np.unique(X[:, node_feat])
                for u in unique:
                    self.__split(X[np.where(X[:, node_feat] == u), ])
        return


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset32.csv', delimiter=';').values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    Y = np.asarray(Y).astype(float)
    Y[Y == 2] = 0.
    tree = DecisionTree()
    print(Y)
