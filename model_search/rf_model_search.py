__author__ = '{Alfonso Aguado Bustillo}'

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class RandomForestModelSearch:

    def __init__(self, partitions, hyper_parameters=None, model=None, y_pred=None):
        self.partitions = partitions  # train, validation and test data partitions
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.y_pred = y_pred  # prediction

    def load_hyper_parameters(self):
        # number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth = [2, 5]
        # minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # method of selecting samples for training each tree
        bootstrap = [True, False]
        # create the hyperparameters
        hyper_parameters = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        self.hyper_parameters = hyper_parameters

    def search_best_regression_model(self):
        self.search_best_model(RandomForestRegressor())

    def search_best_classification_model(self):
        self.search_best_model(RandomForestClassifier())

    def search_best_model(self, estimator):
        # load hyperparameters
        self.load_hyper_parameters()
        # random search of parameters, using 3 fold cross validation,
        # search across 10 different combinations, and use all available cores
        random_models = RandomizedSearchCV(estimator=estimator, param_distributions=self.hyper_parameters, n_iter=10,
                                           cv=5, verbose=2, random_state=0, n_jobs=None)

        self.model = random_models.fit(self.partitions.x_train, self.partitions.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.partitions.x_test)
