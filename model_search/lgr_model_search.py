__author__ = '{Alfonso Aguado Bustillo}'

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class LogisticRegressionModelSearch:

    def __init__(self, partitions, hyper_parameters=None, model=None, y_pred=None):
        self.partitions = partitions  # train, validation and test data partitions
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.y_pred = y_pred  # prediction

    def load_hyper_parameters(self):
        # norm used in the penalization
        penalty = ['l1', 'l2']
        # inverse of regularization, higher values of C implies less regularization
        c = [0.001, .009, 0.01, .09, 1, 5, 10, 25]
        # create the hyper_parameters
        hyper_parameters = {'penalty': penalty, 'c': c}

        self.hyper_parameters = hyper_parameters

    def search_best_model(self):
        # load hyperparameters
        self.load_hyper_parameters()
        estimator = OneVsRestClassifier(LogisticRegression())
        # random search of parameters, using 3 fold cross validation,
        # search across 10 different combinations, and use all available cores
        random_models = RandomizedSearchCV(estimator=estimator, param_distributions=self.hyper_parameters, n_iter=10, cv=5,
                                       verbose=2, random_state=0, n_jobs=None)

        self.model = random_models.fit(self.partitions.x_train, self.partitions.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.partitions.x_test)