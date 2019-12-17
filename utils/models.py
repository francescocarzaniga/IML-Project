import numpy as np
import pandas as pd
from utils.preprocessing import label_to_numerical
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier


class Tree(object):
    def __init__(self, parent=None, children=None, feature=None, threshold=None, direction=None):
        if children is None:
            children = []
        self.parent = parent
        self.children = children
        self.feature = feature
        self.threshold = threshold
        self.direction = direction

    def add_child(self, child):
        self.children.append(child)

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def get_feature(self):
        return self.feature

    def get_threshold(self):
        return self.threshold

    def get_all_features(self):
        node = self
        features_list = []
        while node.parent is not None:
            features_list.append(int(node.get_feature()))
            node = node.parent
        return np.asarray(features_list)

    def get_direction(self):
        return self.direction

    def set_parent(self, parent):
        self.parent = parent

    def set_feature(self, feature):
        self.feature = feature

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_direction(self, direction):
        self.direction = direction

    def __max_depth(self, tree):
        if isinstance(tree, Leaf):
            return 0
        elif len(tree.children) == 0:
            return 0
        else:
            depth = []
            for child in tree.children:
                depth.append(self.__max_depth(child))
            return np.amax(depth)+1.

    def get_max_depth(self):
        return self.__max_depth(self)


class Leaf(object):
    def __init__(self, parent=None, decision=None, direction=None):
        self.parent = parent
        self.decision = decision
        self.direction = direction

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_decision(self):
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def get_direction(self):
        return self.direction

    def set_direction(self, direction):
        self.direction = direction


class DecisionTree(object):
    def __init__(self, max_depth=None, max_features=None, bootstrap=False):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = Tree()
        self.bootstrap = bootstrap
        self._queue = []

    @staticmethod
    def __entropy(labels):
        if labels.size == 0:
            return 0
        unique, counts = np.unique(labels, return_counts=True)
        return np.sum([-counts[i]/np.sum(counts)*np.log2(counts[i]/np.sum(counts)) for i in range(len(unique))])

    def __gain(self, y, subsets):
        entropy_node = self.__entropy(y)
        total_length = np.sum([subset.size for subset in subsets], dtype=np.float64)
        weights = [subset.size/total_length for subset in subsets]
        entropy_child = np.sum([weights[i]*self.__entropy(y[subsets[i]]) for i in range(len(subsets))])
        return entropy_node-entropy_child

    def __split(self, X, y, node, direction=None):
        if node.parent is not None and node not in node.parent.children:
            return
        max_features = self.max_features
        excluded_features = node.get_all_features()
        features = np.delete(np.arange(X.shape[1]), excluded_features)
        if features.size > max_features:
            random_features = np.random.choice(features, max_features, replace=False)
        else:
            random_features = features
        max_gain = 0.
        max_feature = -1
        best_threshold = None
        y_features = np.unique(y)
        if y_features.size == 1:
            leaf = Leaf(parent=node, decision=y_features[0], direction=direction)
            node.add_child(leaf)
            return
        if self.max_depth is not None and self.tree.get_max_depth() >= self.max_depth:
            leaf = Leaf(parent=node, decision=round(float(np.mean(y))), direction=direction)
            node.add_child(leaf)
            return
        for feature in random_features:
            feature_vector = X[:, feature]
            try:
                feature_vector = np.array(feature_vector, dtype=np.float64)
            except ValueError:
                feature_vector = np.array(feature_vector, dtype=object)
            unique = np.unique(feature_vector)
            if feature_vector.dtype == 'object':
                subsets = [np.where(feature_vector == u)[0] for u in unique]
                gain = self.__gain(y, subsets)
                threshold = None
            elif feature_vector.dtype == 'float64':
                threshold_gains = []
                for u in unique:
                    below = np.where(feature_vector <= u)[0]
                    above = np.where(feature_vector > u)[0]
                    threshold_gains.append(self.__gain(y, [below, above]))
                gain = np.nanmax(threshold_gains)
                threshold = unique[np.nanargmax(threshold_gains)]
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
                best_threshold = threshold
        if max_gain == 0.:
            new_node = Leaf(parent=node.parent, decision=round(float(np.mean(y))), direction=node.get_direction())
            substitute = node.parent.children.index(node)
            [self._queue.remove(child) for child in node.children if child in self._queue]
            node.parent.children[substitute] = new_node
            return
        new_node = Tree(parent=node, direction=direction, feature=max_feature, threshold=best_threshold)
        node.add_child(new_node)
        self._queue.append(new_node)
        return

    def __prune(self, X, y):
        return

    def fit(self, X, y):
        if self.bootstrap:
            indices = np.random.choice(y.size, int(self.bootstrap*y.size))
            X = X[indices]
            y = y[indices]
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.__split(X, y, self.tree)
        while len(self._queue) > 0:
            node = self._queue.pop()
            node_feat = node.get_feature()
            node_thresh = node.get_threshold()
            if node_thresh is None:
                unique = np.unique(X[:, node_feat])
                for u in unique:
                    if np.where(X[:, node_feat] == u)[0].size > 0:
                        split_dataset = X[np.where(X[:, node_feat] == u)]
                        split_label = y[np.where(X[:, node_feat] == u)]
                        self.__split(split_dataset, split_label, node, u)
            else:
                if np.where(X[:, node_feat] <= node_thresh)[0].size > 0 and \
                        np.where(X[:, node_feat] > node_thresh)[0].size > 0:
                    dataset_less = X[np.where(X[:, node_feat] <= node_thresh)]
                    dataset_greater = X[np.where(X[:, node_feat] > node_thresh)]
                    label_less = y[np.where(X[:, node_feat] <= node_thresh)]
                    label_greater = y[np.where(X[:, node_feat] > node_thresh)]
                    self.__split(dataset_less, label_less, node, 'l')
                    self.__split(dataset_greater, label_greater, node, 'g')
        return

    def predict(self, X):
        prediction = []
        for sample in X:
            node = self.tree.children[0]
            while not isinstance(node, Leaf):
                feature = node.get_feature()
                threshold = node.get_threshold()
                if threshold is None:
                    value = sample[feature]
                    children_direction = [child.direction for child in node.children]
                    direction = children_direction.index(value)
                    node = node.children[direction]
                else:
                    children_direction = [child.direction for child in node.children]
                    if sample[feature] - threshold < 0:
                        direction = children_direction.index('l')
                    else:
                        direction = children_direction.index('g')
                    node = node.children[direction]
            prediction.append(node.get_decision())
        return np.asarray(prediction)


class RandomForest(object):
    def __init__(self, max_depth=None, max_features=None, n_estimators=10, bootstrap=0.5, n_jobs=-1):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self._estimators = []

    def __make_estimators(self):
        estimators = Parallel(n_jobs=self.n_jobs)(delayed(DecisionTree)
                                                  (max_depth=self.max_depth, max_features=self.max_features,
                                                   bootstrap=self.bootstrap) for i in range(self.n_estimators))
        return estimators

    @staticmethod
    def __parallel_build_trees(tree, X, y):
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        estimators = self.__make_estimators()
        result = Parallel(n_jobs=self.n_jobs)(delayed(self.__parallel_build_trees)(tree, X, y) for tree in estimators)
        self._estimators = result
        return

    def predict(self, X):
        results = Parallel(n_jobs=self.n_jobs)(delayed(element.predict)(X) for element in self._estimators)
        return np.stack(results).mean(axis=0).round()


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset32.csv', delimiter=';').drop('vehicle_number', axis=1).values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    X = label_to_numerical(X)
    X[np.isnan(X)] = 0.
    Y = np.asarray(Y).astype(float)
    Y[Y == 2] = 0.
    # forest = RandomForest(n_estimators=8)
    # forest.fit(X, Y)
    # print(forest.predict(X))
    # print(np.mean(abs(forest.predict(X)-Y)))
    test = DecisionTreeClassifier(criterion='entropy')
    test.fit(X, Y)
    print(test.predict(X))
    print(np.mean(abs(test.predict(X) - Y)))
