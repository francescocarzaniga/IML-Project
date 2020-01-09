import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier

#DATA PREPROCESSING
dataset = pd.read_csv('dataset32.csv', delimiter = ";").values

y = dataset[:,13] #'class'

unique, counts = np.unique(y, return_counts=True)
print([counts[i]/np.sum(counts) for i in range(len(counts))]) #sono bilanciati!

y = y.astype(np.float).reshape((dataset.shape[0],1))

dataset = dataset[:,[0,1,2,3,4,5,6,7,8,9,10,12]] #selection of features

#print(dataset.shape)

le = preprocessing.LabelEncoder()
dataset[:,3] = le.fit_transform(dataset[:,3]) # 'road_state': average = 0, bad = 1, good = 2
dataset[:,11] = le.fit_transform(dataset[:,11]) # 'road_type': local = 0 , national = 1, regional = 2

dataset = np.asarray(dataset, dtype=np.float64)

# dataset has 10 NaN values
for i in range(12):
    if any(np.isnan(dataset[:,i])):
       print("Feature", i, "has", sum(np.isnan(dataset[:,i])), "NaN value(s)")

#substitute the missing values by the mean value of the feature
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset = imp_mean.fit_transform(dataset)

#shuffling
def shuffle(dataset, y):
    z = np.hstack((dataset, y))
    np.random.shuffle(z)
    return np.hsplit(z, [dataset.shape[1]])

dataset, y = shuffle(dataset, y)

#print(y)
#print(y.ravel())
#print(y.ravel().)

#print(y)
#print(y.ravel())
#print(y.ravel().tolist())
#print(type(y.ravel().tolist()))
# def most_frequent(List):
#     occurrence_count = Counter(List)
#     return occurrence_count.most_common(1)[0][0]
#
# popular_s = most_frequent(y.ravel().tolist())
#
# print(int(popular_s))

#splitting
# def splitting(x, y, test_size=0.2):
#     n = x.shape[0]
#     train_size = int(n * (1 - test_size))
#     return x[:train_size, ], x[train_size:, ], y[:train_size, ], y[train_size:, ]
#
# x_train, x_test, y_train, y_test = splitting(dataset, y)

# print("x_train: ", x_train.shape)
# print("y_train: ", y_train.shape)
# print("x_test: ", x_test.shape)
# print("y_test: ", y_test.shape)

## CROSS VALIDATION
# class Transform(object):  # class to transform the multiclass problem to 1vs.all problem
#     def __init__(self, model=None, **parameters):
#         self.model = model
#         self.model_list = []
#         self.classes = None
#         self.parameters = parameters
#        # self.popular = None
#
#     def get_params(self, deep=True):
#         return {**{"model": self.model}, **self.parameters}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#     #def most_frequent(self, List):
#         #occurrence_count = Counter(List)
#         #return occurrence_count.most_common(1)[0][0]
#
#     def fit(self, X, y):
#         classes = np.unique(y)
#         self.classes = classes
#         # self.popular = self.most_frequent(y.ravel().tolist())
#         for item in classes:
#             y_mod = np.copy(y)
#             actual_model = self.model().set_params(**self.parameters)
#             y_mod[y_mod != item] = classes[(np.where(classes == item)[0]+1) % classes.size] # to obtain 1vs.all
#             # (da correggere) y_mod = np.place(y_mod, y_mod != item, classes[(np.where(classes == item)[0]+1) % classes.size])
#             actual_model.fit(X, y_mod)
#             self.model_list.append(actual_model)
#         return
#
#     def predict(self, X):
#         predict_array = np.stack([model.predict(X) for model in self.model_list])
#         # DA FARE predict_prob =
#         val = []
#         #for i in range(self.classes.size):
#         # for k in range(X.shape[0]):
#         #     if predict_array[0,k] == self.classes[0]:
#         #         val.append(self.classes[0])
#         #
#         #     elif predict_array[(i+1) % self.classes.size,k] == self.classes[(i+1) % self.classes.size]:
#         #         val.append(self.classes[(i+1) % self.classes.size])
#         #     else:
#         #         val.append(self.classes[(i+2) % self.classes.size])
#
#         # for k in range(X.shape[0]):
#         #     index = np.where(predict_array[:,k] == self.classes)
#         #     if index[0].size == 1:
#         #         val.append(index[0][0])
#         #     else:
#         #         val.append(self.most_frequent(self.popular))
#
#         val_array = np.asarray(val)
#         return val_array
#
#     def score(self, X, y):
#         label_predict = self.predict(X)
#         loss = np.mean(y.ravel() == label_predict)
#         return loss

# model_LinearRegression = linear_model.LinearRegression
# transformed_LinearRegression = Transform(model_LinearRegression)
# transformed_LinearRegression.fit(dataset, y.ravel())
# transformed_LinearRegression.predict(dataset)
# val_LinearRegression = cross_validate(transformed_LinearRegression, dataset, y.ravel(), cv=5)
# print(val_LinearRegression)

# model_SVC = LinearSVC
# transformed_SVC = Transform(model_SVC)
# transformed_SVC.fit(dataset, y.ravel())
# transformed_SVC.predict(dataset)
# val_SVC = cross_validate(transformed_SVC, dataset, y.ravel(), cv=5)

# OnevsAll
class OnevsAll(object):
    def __init__(self, model=None, n_jobs=-1, **parameters):  # initialize self
        self.model = model
        self.model_list = []
        self.n_jobs = n_jobs
        self.classes = None
        self.parameters = parameters
        self.popular = None

    def get_params(self, deep=True):  # get parameters
        return {**{"model": self.model}, **{"n_jobs": self.n_jobs}, **self.parameters}

    def set_params(self, **parameters):  # set parameters
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @staticmethod
    def most_frequent(y):
        occurrence_count = Counter(y)
        return occurrence_count.most_common(1)[0][0]

    def __fit_ova_estimator(self, X, y, class_one):  # trasform all the models into 1 vs 0 (class_one is 1, the rest 0)
        current_model = self.model().set_params(**self.parameters)
        y_binarized = np.zeros_like(y)
        y_binarized[y == class_one] = 1
        y_binarized[y != class_one] = 0
        current_model.fit(X, y_binarized)
        return current_model, class_one

    def fit(self, X, y):
        self.classes = np.unique(y)
        models = Parallel(n_jobs=self.n_jobs)(delayed(self.__fit_ova_estimator)
                                              (X, y, self.classes[i]) for i in
                                              range(len(self.classes)))
        self.model_list = list(zip(*models))
        self.popular = self.most_frequent(y.ravel())
        return

    # def fit(self, X, y): # fit the three models
    # classes = np.unique(y)
    # self.classes = classes
    # for item in classes:
    # y_mod = np.copy(y)
    # actual_model = self.model().set_params(**self.parameters)
    # y_mod[y_mod != item] = classes[(np.where(classes == item)[0]+1) % classes.size]
    # model0: 0vs1, model1:1vs2, model2:2vs0
    # actual_model.fit(X, y_mod)
    # self.model_list.append(actual_model)
    # return

    @staticmethod
    def __predict_ova_estimator(X, model):
        return model.predict(X)

    @staticmethod
    def __predict_proba_ova_estimator(X, model):
        try:
            confidence = np.max(model.predict_proba(X), axis=1)
        except (AttributeError, NotImplementedError):
            confidence = model.decision_function(X)
        return confidence

    def predict(self, X):
        # predict from a certain model the best class for every label in X
        # if there are possible missunderstanding between the models (two models give an appropriate value, e.g. model 0 gives 0 and model 1 gives 1),
        # the function take the class given by the most confident model

        models = self.model_list[0]
        predictions = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_ova_estimator)(X, models[i])
                                                            for i in range(len(models)))).astype(dtype=np.int32)
        confidences = np.stack(Parallel(n_jobs=self.n_jobs)(delayed(self.__predict_proba_ova_estimator)(X, models[i])
                                                            for i in range(len(models))))

        # predict_array = np.stack([model.predict(X) for model in self.model_list]) #every model correspond to a row
        # confidences_array = np.stack([_predict_proba_ova_estimator(X,model) for model in self.model_list])

        val = []
        i = 0
        for k in range(X.shape[0]):
            index = np.argwhere(predictions[:, k] == 1)
            if index.size == 1:  # if there is a unique 1 in the column k
                val.append(self.classes[index])
            elif index.size == 0:  # if there are none #TO DO
                conf = confidences[:,k]
                val.append(self.classes[np.argmin(conf)])
                i += 1
            else:
                conf = np.multiply((predictions[:, k] + confidences[:, k]), (predictions[:, k]))  # add the confidence only to the values with 1
                val.append(self.classes[np.argmax(conf)])
        val_array = np.asarray(val)
        print(i)
        return val_array

    def score(self, X, y):  # this function return the accuracy of the prediction given X and y
        label_predict = self.predict(X)
        loss = np.mean(y.ravel() == label_predict)
        return loss
# SVM
#model_SVM = svm.SVC
#transformed_SVM = OnevsAll(model_SVM, kernel='poly')
# # #transformed_svm.fit(dataset,y.ravel())
# # #transformed_svm.predict(dataset)
#val_SVM = cross_validate(transformed_SVM, dataset, y.ravel(), cv=5)
#print(val_SVM)
# #
# clf = OneVsRestClassifier(model_SVM(kernel='poly', gamma='auto'))
# val_1 = cross_validate(clf, dataset, y.ravel(), cv=5)
# print(val_1)

# SVM = model_SVM()
# SVM.fit(dataset, y.ravel())

# Kernelized SVM
# model_rbf = svm.SVC
# transformed_rbf = Transform(model_rbf, kernel='rbf')
# # transformed_rbf.fit(dataset, y.ravel())
# # transformed_rbf.predict(dataset)
# val_rbf = cross_validate(transformed_rbf, dataset, y.ravel(), cv=5)
# print(val_rbf)

#K-nearest neighbour algorithm
#model_K = KNeighborsClassifier
#estimator = Pipeline([("imputer", SimpleImputer(missing_values= np.nan,strategy="median")),("Transform",OnevsAll(model_K,n_neighbors=5))])
#val_K = cross_validate(estimator, dataset, y.ravel(), cv=5)
#print(val_K)


#clf = OneVsRestClassifier(model_K())
#val_1 = cross_validate(clf, dataset, y.ravel(), cv=5)
#print(val_1)

##LINEAR REGRESSION

# x_train = np.hstack((np.ones((x_train.shape[0],1)), x_train))
# x_test = np.hstack((np.ones((x_test.shape[0],1)), x_test))
#
# # print(x_train.shape)
# # print(x_test.shape)
#
# def my_error(y_true, y_pred):
#     return (1/len(y_true))*sum((y_true - y_pred)**2) #empirical error function (squared L^2 norm)
#
# lr = linear_model.LinearRegression(fit_intercept=False, normalize=False).fit(x_train, y_train)
# beta_sklearn = np.transpose(lr.coef_)
#
# def linear_regression_predict(x, beta):
#     result = x.dot(beta)
#     return result
#
# print("Training error: ", my_error(y_train, linear_regression_predict(x_train, beta_sklearn)))
# print("Test error: ", my_error(y_test, linear_regression_predict(x_test, beta_sklearn)))

# plt.plot(y_test - linear_regression_predict(x_test, beta_sklearn), 'ro')
# plt.show()




