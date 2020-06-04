__author__ = '{Alfonso Aguado Bustillo}'

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC, SVR
from sklearn.multiclass import OneVsRestClassifier


class SVM:
    def __init__(self, partitions, hyper_parameters=None, model=None, y_pred=None):
        self.partitions = partitions  # train, validation and test data partitions
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.y_pred = y_pred  # prediction

    def load_hyper_parameters(self):
        # kernel used
        kernel = ('linear', 'rbf')
        # inverse of regularization, higher values of C implies less regularization
        c = [1, 10]
        # create the hyper_parameters
        hyper_parameters = {'kernel': kernel, 'c': c}
        self.hyper_parameters = hyper_parameters

    def search_best_regression_model(self):
        self.search_best_model(SVR())

    def search_best_classification_model(self):
        self.search_best_model(OneVsRestClassifier(LinearSVC()))

    def search_best_model(self, estimator):
        # load hyperparameters
        self.load_hyper_parameters()
        # random search of parameters, using 3 fold cross validation,
        # search across 10 different combinations, and use all available cores
        random_models = RandomizedSearchCV(estimator=estimator, param_distributions=self.hyper_parameters, n_iter=10, cv=5,
                                       verbose=2, random_state=0, n_jobs=None)

        self.model = random_models.fit(self.partitions.x_train, self.partitions.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.partitions.x_test)