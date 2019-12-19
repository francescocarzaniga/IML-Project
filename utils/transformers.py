from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from utils.preprocessing import label_to_numerical, impute_whole
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class OneVsOne(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, model=None, n_jobs=-1, **parameters):
        self.model = model
        self.n_jobs = n_jobs
        self.parameters = parameters
        self.classes = None
        self.model_list = None

    def __fit_ovo_estimator(self, X, y, class_one, class_two):
        class_selection = np.logical_or(y == class_one, y == class_two)
        current_model = self.model().set_params(**self.parameters)
        y = y[class_selection]
        X = X[class_selection]
        current_model.fit(X, y)
        return current_model, class_one, class_two

    def fit(self, X, y):
        self.classes = np.unique(y)
        models = Parallel(n_jobs=self.n_jobs)(delayed(self.__fit_ovo_estimator)
                                              (X, y, self.classes[i], self.classes[j]) for i in range(len(self.classes))
                                              for j in range(i + 1, len(self.classes)))
        self.model_list = list(zip(*models))
        return

    @staticmethod
    def __predict_ovo_estimator(X, model):
        return model.predict(X)

    @staticmethod
    def __predict_proba_ovo_estimator(X, model):
        try:
            confidence = model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            confidence = model.decision_function(X)
        return confidence

    def __compute_confidence(self, unique, max_votes, predictions, confidence, i, kind):
        sample_confidence = np.zeros(self.classes.size)
        if kind:
            for k in range(len(self.model_list[0])):
                sample_confidence[np.argwhere(self.model_list[1][k] == self.classes)] += confidence[i, 0, k]
                sample_confidence[np.argwhere(self.model_list[2][k] == self.classes)] += confidence[i, 1, k]
            for c in np.flip(np.sort(sample_confidence)):
                if self.classes[np.argwhere(sample_confidence == c)] in unique[max_votes]:
                    return self.classes[np.argwhere(sample_confidence == c)]
        else:
            for k in range(len(self.model_list[0])):
                sample_confidence[np.argwhere(predictions[i, k] == self.classes)] += np.abs(confidence[k, i])
            for c in np.sort(sample_confidence):
                if self.classes[np.argwhere(sample_confidence == c)] in unique[max_votes]:
                    return self.classes[np.argwhere(sample_confidence == c)]

    def __compute_final_prediction(self, predictions, confidence, i, kind):
        unique, counts = np.unique(predictions[i], return_counts=True)
        max_votes = np.argwhere(counts == np.max(counts))
        if max_votes.size > 1:
            return self.__compute_confidence(unique, max_votes, predictions, confidence, i, kind=kind)
        else:
            return unique[max_votes]

    def predict(self, X):
        models = self.model_list[0]
        predictions = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_ovo_estimator)(X, models[i])
                                                            for i in range(len(models)))).astype(dtype=np.int32).T
        confidence = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_proba_ovo_estimator)(X, models[i])
                                                           for i in range(len(models))))
        kind = 0
        if len(confidence.shape) > 2:
            kind = 1
        total_predictions = Parallel(n_jobs=self.n_jobs)(delayed(self.__compute_final_prediction)
                                                         (predictions, confidence, i, kind) for i in range(X.shape[0]))
        return np.asarray(total_predictions).ravel()


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset32.csv', delimiter=';').drop('vehicle_number', axis=1).values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    Y = np.asarray(Y).astype(float)
    X = impute_whole(X)
    X = label_to_numerical(X)
    dataset_train, dataset_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                                            random_state=42)
    estimator = OneVsOne(RandomForestClassifier)
    estimator.fit(dataset_train, label_train)
    print(estimator.score(dataset_test, label_test))
    test = RandomForestClassifier()
    test.fit(dataset_train, label_train)
    print(test.score(dataset_test, label_test))
