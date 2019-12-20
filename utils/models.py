import numpy as np
import pandas as pd
from utils.preprocessing import label_to_numerical, impute_whole
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from time import perf_counter


class Tree(object):
    def __init__(self, parent=None, children=None, feature=None, threshold=None, direction=None, excluded_samples=None,
                 is_leaf=False, decision=None, confidence=None):
        if children is None:
            children = []
        self.parent = parent
        self.children = children
        self.feature = feature
        self.threshold = threshold
        self.direction = direction
        self.excluded_samples = excluded_samples
        self.is_leaf = is_leaf
        self.decision = decision
        self.confidence = confidence
        self.depth = self.compute_depth()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def compute_depth(self):
        depth = 0
        node = self
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth

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
        while node is not None:
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
        if tree.is_leaf:
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

    def get_depth(self):
        return self.depth

    def set_excluded_samples(self, excluded_samples):
        self.excluded_samples = excluded_samples
        return

    def get_all_excluded_samples(self):
        node = self
        samples_list = np.asarray([])
        while node is not None:
            samples_list = np.concatenate([samples_list, node.get_excluded_samples().ravel()])
            node = node.parent
        return np.asarray(samples_list)

    def get_is_leaf(self):
        return self.is_leaf

    def set_is_leaf(self, is_leaf):
        self.is_leaf = is_leaf

    def get_decision(self):
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def get_excluded_samples(self):
        return np.asarray(self.excluded_samples)

    def get_confidence(self):
        return self.confidence


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None
        self._queue = []
        self.classes = None

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

    def __split(self, X, y, node, excluded_samples=None, direction=None):
        # Choose remaining features and samples to be tested
        dataset_size, label_size = X.shape
        if excluded_samples is None:
            excluded_samples = []
        if node is not None:
            excluded_features = node.get_all_features()
            features = np.delete(np.arange(label_size), excluded_features)
            all_excluded_samples = node.get_all_excluded_samples()
            all_excluded_samples = np.concatenate([all_excluded_samples, excluded_samples]).astype(dtype=np.int32)
            samples = np.delete(np.arange(dataset_size), all_excluded_samples)
        else:
            features = np.arange(label_size)
            samples = np.arange(dataset_size)
        y_orig = np.copy(y)
        X = X[samples]
        y = y[samples]
        classes, counts = np.unique(y, return_counts=True)
        confidence = np.zeros(2)
        # Base case 1, labels are all the same so create leaf where decision is label
        if classes.size == 1:
            confidence[np.argwhere(self.classes == classes[0])] = 1.
            leaf = Tree(parent=node, decision=classes[0], direction=direction, is_leaf=True, confidence=confidence)
            node.add_child(leaf)
            return 1
        # Base case 2, no labels associated to this class so create failure decision (should never happen)
        elif classes.size == 0:
            unique, counts = np.unique(y_orig[samples], return_counts=True)
            total_labels = np.sum(counts)
            for u in range(unique.size):
                confidence[np.argwhere(self.classes == unique[u])] = counts[u]/total_labels
            all_excluded_samples = node.get_all_excluded_samples().astype(dtype=np.int32)
            samples = np.delete(np.arange(dataset_size), all_excluded_samples)
            leaf = Tree(parent=node, decision=int(np.median(y_orig[samples]).round()), direction=direction,
                        is_leaf=True, confidence=confidence)
            node.add_child(leaf)
            return 2
        # Max depth parameter must be respected
        if self.max_depth is not None and node is not None and node.get_max_depth() == self.max_depth - 1:
            unique, counts = np.unique(y, return_counts=True)
            total_labels = np.sum(counts)
            for u in range(unique.size):
                confidence[np.argwhere(self.classes == unique[u])] = counts[u]/total_labels
            leaf = Tree(parent=node, decision=int(np.median(y).round()), direction=direction, is_leaf=True,
                        confidence=confidence)
            node.add_child(leaf)
            return 4
        # Max_features must be respected
        max_features = self.max_features
        if max_features is not None and features.size > max_features:
            random_features = np.random.choice(features, max_features, replace=False)
        else:
            random_features = features
        # Try all the chosen features
        max_gain = 0.
        max_feature = -1
        best_threshold = None
        for feature in random_features:
            feature_vector = X[:, feature]
            try:
                feature_vector = np.array(feature_vector, dtype=np.float64)
            except ValueError:
                feature_vector = np.array(feature_vector, dtype=object)
            unique, counts = np.unique(feature_vector, return_counts=True)
            if feature_vector.dtype == 'object':
                subsets = [np.argwhere(feature_vector == u) for u in unique]
                gain = self.__gain(y, subsets)
                threshold = None
            elif feature_vector.dtype == 'float64':
                threshold_gains = []
                for u in unique:
                    below = np.argwhere(feature_vector <= u)
                    above = np.argwhere(feature_vector > u)
                    threshold_gains.append(self.__gain(y, [below, above]))
                gain = np.nanmax(threshold_gains)
                threshold = unique[np.nanargmax(threshold_gains)]
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
                best_threshold = threshold
        # Base case 3
        if max_gain == 0.:
            unique, counts = np.unique(y_orig[samples], return_counts=True)
            total_labels = np.sum(counts)
            for u in range(unique.size):
                confidence[np.argwhere(self.classes == unique[u])] = counts[u] / total_labels
            all_excluded_samples = node.get_all_excluded_samples().astype(dtype=np.int32)
            samples = np.delete(np.arange(dataset_size), all_excluded_samples)
            new_node = Tree(parent=node.parent, decision=int(np.median(y_orig[samples]).round()),
                            direction=node.get_direction(), is_leaf=True, confidence=confidence)
            substitute = node.parent.children.index(node)
            node.parent.children[substitute] = new_node
            return 3
        # Create new node with best feature
        new_node = Tree(parent=node, direction=direction, feature=max_feature, threshold=best_threshold,
                        excluded_samples=excluded_samples)
        if node is not None:
            node.add_child(new_node)
        return new_node

    def __prune(self, X, y):
        return

    def __create_nodes_numerical(self, X, y, feature_vector, node_thresh, node):
        less = np.argwhere(feature_vector <= node_thresh).ravel()
        great = np.argwhere(feature_vector > node_thresh).ravel()
        case = self.__split(X, y, node, great, 'l')
        if isinstance(case, Tree):
            self._queue.append(case)
        elif case == 3:
            return
        case = self.__split(X, y, node, less, 'g')
        if isinstance(case, Tree):
            self._queue.append(case)
        elif case == 3:
            return

    def __create_nodes_categorical(self, X, y, feature_vector, unique, node):
        for u in unique:
            excluded_samples = np.argwhere(feature_vector != u).ravel()
            case = self.__split(X, y, node, excluded_samples, u)
            if isinstance(case, Tree):
                self._queue.append(case)
            elif case == 3:
                return

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.classes = np.unique(y)
        self.tree = self.__split(X, y, self.tree)
        self._queue.append(self.tree)
        while len(self._queue) > 0:
            node = self._queue.pop()
            node_feat = node.get_feature()
            node_thresh = node.get_threshold()
            feature_vector = X[:, node_feat]
            unique, counts = np.unique(feature_vector, return_counts=True)
            if node_thresh is None:
                self.__create_nodes_categorical(X, y, feature_vector, unique, node)
            else:
                self.__create_nodes_numerical(X, y, feature_vector, node_thresh, node)
        return

    def predict(self, X):
        prediction = []
        for sample in X:
            node = self.tree
            while not node.is_leaf:
                feature = node.get_feature()
                threshold = node.get_threshold()
                if threshold is None:
                    value = sample[feature]
                    children_direction = [child.direction for child in node.children]
                    direction = children_direction.index(value)
                    node = node.children[direction]
                else:
                    if sample[feature] - threshold < 0:
                        direction = 0
                    else:
                        direction = 1
                    node = node.children[direction]
            prediction.append(node.get_decision())
        return np.asarray(prediction)

    def predict_proba(self, X):
        proba = []
        for sample in X:
            node = self.tree
            while not node.is_leaf:
                feature = node.get_feature()
                threshold = node.get_threshold()
                if threshold is None:
                    value = sample[feature]
                    children_direction = [child.direction for child in node.children]
                    direction = children_direction.index(value)
                    node = node.children[direction]
                else:
                    if sample[feature] - threshold < 0:
                        direction = 0
                    else:
                        direction = 1
                    node = node.children[direction]
            proba.append(node.get_confidence())
        return np.asarray(proba)


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, max_features=None, n_estimators=10, bootstrap=1., n_jobs=-1):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self._estimators = []

    def __make_estimators(self):
        estimators = Parallel(n_jobs=self.n_jobs)\
            (delayed(DecisionTree)(max_depth=self.max_depth, max_features=self.max_features)
             for i in range(self.n_estimators))
        return estimators

    @staticmethod
    def __parallel_build_trees(tree, X, y, bootstrap):
        if bootstrap:
            samples = np.random.choice(np.arange(X.shape[0]), int(bootstrap*X.shape[0]))
            X = X[samples]
            y = y[samples]
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        estimators = self.__make_estimators()
        result = Parallel(n_jobs=self.n_jobs)\
            (delayed(self.__parallel_build_trees)(tree, X, y, self.bootstrap) for tree in estimators)
        self._estimators = result
        return

    def predict(self, X):
        results = Parallel(n_jobs=self.n_jobs)(delayed(element.predict)(X) for element in self._estimators)
        return np.median(np.stack(results), axis=0).round()

    def predict_proba(self, X):
        results = Parallel(n_jobs=self.n_jobs)(delayed(element.predict_proba)(X) for element in self._estimators)
        return np.mean(np.stack(results, axis=2), axis=2)


def get_leaf_decisions(tree, leaf_decisions):
    if tree.is_leaf:
        leaf_decisions.append(tree.decision)
    elif len(tree.children) == 0:
        return 0
    else:
        for child in tree.children:
            get_leaf_decisions(child, leaf_decisions)


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset32.csv', delimiter=';').drop('vehicle_number', axis=1).values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    # X = label_to_numerical(X)
    # X[np.isnan(X)] = 0.
    X = impute_whole(X)
    Y = np.asarray(Y).astype(float)
    Y[Y == 2] = 0.
    dataset_train, dataset_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                                            random_state=42)
    start = perf_counter()
    forest = RandomForest()
    forest.fit(dataset_train, label_train)
    # forest.predict_proba(dataset_test)
    print(forest.score(dataset_test, label_test))
    print(perf_counter()-start)
    X = label_to_numerical(X)
    dataset_train, dataset_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                                            random_state=42)
    start = perf_counter()
    sk_tree = RandomForestClassifier(criterion='entropy')
    sk_tree.fit(dataset_train, label_train)
    # print(sk_tree.score(dataset_train, label_train))
    print(sk_tree.score(dataset_test, label_test))
    print(perf_counter()-start)

