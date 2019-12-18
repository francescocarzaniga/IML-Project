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
        self.depth = self.compute_depth()

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

    def get_depth(self):
        return self.depth


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
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None
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
        classes = np.unique(y)
        # Base case 1, labels are all the same so create leaf where decision is label
        if classes.size == 1:
            leaf = Leaf(parent=node, decision=classes[0], direction=direction)
            node.add_child(leaf)
            return 1
        # Base case 2, no labels associated to this class so create failure decision (should never happen)
        elif classes.size == 0:
            leaf = Leaf(parent=node, decision='failure', direction=direction)
            node.add_child(leaf)
            return 2
        # Max depth parameter must be respected
        if self.max_depth is not None and node.get_max_depth() == self.max_depth - 1:
            leaf = Leaf(parent=node, decision=int(np.mean(y).round()), direction=direction)
            node.add_child(leaf)
            return 4
        # Choose remaining features to be tested
        if node is not None:
            excluded_features = node.get_all_features()
            features = np.delete(np.arange(X.shape[1]), excluded_features)
        else:
            features = np.arange(X.shape[1])
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
            # Substitute nan with most frequent value
            counts, unique = np.unique(feature_vector, return_counts=True)
            feature_vector[np.isnan(feature_vector)] = unique[np.argmax(counts)]
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
        # Base case 3
        if max_gain == 0.:
            new_node = Leaf(parent=node.parent, decision=int(np.mean(y).round()), direction=node.get_direction())
            substitute = node.parent.children.index(node)
            node.parent.children[substitute] = new_node
            return 3
        # Create new node with best feature
        new_node = Tree(parent=node, direction=direction, feature=max_feature, threshold=best_threshold)
        if node is not None:
            node.add_child(new_node)
        self._queue.append(new_node)
        return new_node

    def __prune(self, X, y):
        return

    def __create_nodes_numerical(self, X, y, feature_vector, node_thresh, node):
        if np.where(feature_vector <= node_thresh)[0].size > 0 and \
                np.where(feature_vector > node_thresh)[0].size > 0:
            dataset_less = X[feature_vector <= node_thresh]
            dataset_greater = X[feature_vector > node_thresh]
            label_less = y[feature_vector <= node_thresh]
            label_greater = y[feature_vector > node_thresh]
            case = self.__split(dataset_less, label_less, node, 'l')
            if isinstance(case, DecisionTree):
                self._queue.append(case)
            elif case == 3:
                return
            case = self.__split(dataset_greater, label_greater, node, 'g')
            if isinstance(case, DecisionTree):
                self._queue.append(case)
            elif case == 3:
                return

    def __create_nodes_categorical(self, X, y, feature_vector, unique, node):
        for u in unique:
            if np.where(feature_vector == u)[0].size > 0:
                split_dataset = X[feature_vector == u]
                split_label = y[feature_vector == u]
                case = self.__split(split_dataset, split_label, node, u)
                if isinstance(case, DecisionTree):
                    self._queue.append(case)
                elif case == 3:
                    return

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree = self.__split(X, y, self.tree)
        while len(self._queue) > 0:
            node = self._queue.pop()
            node_feat = node.get_feature()
            node_thresh = node.get_threshold()
            feature_vector = X[:, node_feat]
            counts, unique = np.unique(feature_vector, return_counts=True)
            feature_vector[np.isnan(feature_vector)] = unique[np.argmax(counts)]
            unique = np.unique(feature_vector)
            if node_thresh is None:
                self.__create_nodes_categorical(X, y, feature_vector, unique, node)
            else:
                self.__create_nodes_numerical(X, y, feature_vector, node_thresh, node)
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


def get_leaf_decisions(tree, leaf_decisions):
    if isinstance(tree, Leaf):
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
    X = label_to_numerical(X)
    X[np.isnan(X)] = 0.
    Y = np.asarray(Y).astype(float)
    Y[Y == 2] = 0.
    unique, counts = np.unique(Y, return_counts=True)
    print([count/np.sum(counts) for count in counts])
    # Y_new = np.copy(Y)
    # Y_new[Y == 0] = 1
    # Y_new[Y == 1] = 0
    # Y = Y_new
    forest = DecisionTree()
    forest.fit(X, Y)
    leaf_decisions = []
    get_leaf_decisions(forest.tree, leaf_decisions)
    print(leaf_decisions)
    unique, counts = np.unique(leaf_decisions, return_counts=True)
    print([count/np.sum(counts) for count in counts])
    # print(forest.predict(X))
    # print(np.mean(abs(forest.predict(X)-Y)))
    # sk_tree = DecisionTreeClassifier()
    # sk_tree.fit(X, Y)
    # print(np.mean(abs(sk_tree.predict(X)-Y)))

