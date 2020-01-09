from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from utils.preprocessing import label_to_numerical, impute_whole
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import perf_counter
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from utils.models import RandomForest
from sklearn.multiclass import OneVsOneClassifier


class OneVsOne(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, model=None, n_jobs=-1, **parameters):
        self.model = model
        self.n_jobs = n_jobs
        self.parameters = parameters
        self.classes = None
        self.model_list = None

    def get_params(self):
        return {**{"model": self.model}, **{"n_jobs": self.n_jobs}, **self.parameters}

    def __fit_ovo_estimator(self, X, y, class_one, class_two):
        class_selection = np.logical_or(y == class_one, y == class_two)
        current_model = self.model().set_params(**self.parameters)
        y = y[class_selection]
        y_binarized = np.zeros_like(y)
        y_binarized[y == class_one] = 0
        y_binarized[y == class_two] = 1
        X = X[class_selection]
        current_model.fit(X, y_binarized)
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
            confidence = np.max(model.predict_proba(X), axis=1)
        except (AttributeError, NotImplementedError):
            confidence = model.decision_function(X)
        return confidence

    def predict(self, X):
        models = self.model_list[0]
        predictions = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_ovo_estimator)(X, models[i])
                                                            for i in range(len(models)))).astype(dtype=np.int32).T
        confidences = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_proba_ovo_estimator)(X, models[i])
                                                            for i in range(len(models)))).T
        votes = np.zeros((X.shape[0], self.classes.size))
        total_confidences = np.zeros_like(votes)
        for model in range(len(models)):
            class_one_m = self.model_list[1][model]
            class_two_m = self.model_list[2][model]
            votes[predictions[:, model] == 0, np.argwhere(self.classes == class_one_m)[0]] += 1
            votes[predictions[:, model] == 1, np.argwhere(self.classes == class_two_m)[0]] += 1
            total_confidences[predictions[:, model] == 0, np.argwhere(self.classes == class_one_m)[0]] += \
                confidences[predictions[:, model] == 0, model]
            total_confidences[predictions[:, model] == 1, np.argwhere(self.classes == class_two_m)[0]] += \
                confidences[predictions[:, model] == 1, model]
        transformed_confidences = (total_confidences /
                                   (3 * (np.abs(total_confidences) + 1)))
        winners = self.classes[np.argmax(votes+transformed_confidences, axis=1)]
        return winners


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset32.csv', delimiter=';').drop('vehicle_number', axis=1).values
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    Y = np.asarray(Y).astype(float)
    X = impute_whole(X)
    X = label_to_numerical(X)
    dataset_train, dataset_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                                            random_state=42)
    fitter = SVC
    start = perf_counter()
    estimator = OneVsOne(SVC, decision_function_shape='ovr')
    estimator.fit(dataset_train, label_train)
    print(estimator.score(dataset_test, label_test))
    print(perf_counter()-start)