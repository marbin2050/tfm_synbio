__author__ = '{Alfonso Aguado Bustillo}'

import numpy as np
import tensorflow as tf
import time
import pandas as pd
import evaluation


class BaseModel:
    def __init__(self, partitions, estimator):
        self.partitions = partitions  # train, validation and test data partitions
        self.model = estimator
        self.prediction = None
        self.results = {}
        self.fitting_time = None

    def run(self):
        start = time.time()
        self.fit()
        end = time.time()
        self.fitting_time = end-start
        self.predict()
        self.evaluate()

    def fit(self):
        self.model = self.model.fit(self.partitions.x_train, self.partitions.y_train)

    def predict(self):
        self.prediction = self.model.predict(self.partitions.x_test)


class Regressor(BaseModel):
    def evaluate(self):
        self.results = evaluation.evaluate_regressor(self.partitions, self.prediction, self.fitting_time)


class Classifier(BaseModel):
    def evaluate(self):
        self.results = evaluation.evaluate_classifier(self.partitions, self.prediction, self.fitting_time)


class ConvolutionalNeuralNetwork:
    def __init__(self, partitions):
        self.partitions = partitions  # train, validation and test data partitions
        self.model = None
        self.prediction = None  # prediction
        self.results = {}
        self.fitting_time = None

    def run_regression(self):
        self.build_regression_model()
        start = time.time()
        self.fit()
        end = time.time()
        self.fitting_time = end - start
        self.predict()
        self.results = evaluation.evaluate_regressor(self.partitions, self.prediction, self.fitting_time)

    def run_classification(self):
        self.build_classification_model()
        start = time.time()
        self.fit()
        end = time.time()
        self.fitting_time = end - start
        self.predict()
        # get discrete values for prediction
        self.prediction = np.where(self.prediction < 0.5, 0., 1.)  # only for classification
        self.results = evaluation.evaluate_classifier(self.partitions, self.prediction, self.fitting_time)

    def build_regression_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(256, 5, activation='relu', input_shape=(95, 1)),
            tf.keras.layers.AveragePooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1),
        ])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        self.model = model

    def build_classification_model(self):
        model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv1D(64, 5, activation='relu', input_shape=(95, 1)),
                    tf.keras.layers.AveragePooling1D(2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(23, activation='sigmoid')
                ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def fit(self):
        EPOCHS = 100
        # the patience parameter is the amount of epochs to check for improvement
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(np.expand_dims(self.partitions.x_train, axis=2), self.partitions.y_train,
                       epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop], verbose=0)

    def predict(self):
        self.prediction = self.model.predict(np.expand_dims(self.partitions.x_test, axis=2), batch_size=10, verbose=0)


class MultilayerPerceptron:
    def __init__(self, partitions):
        self.partitions = partitions  # train, validation and test data partitions
        self.model = None
        self.prediction = None
        self.results = {}
        self.fitting_time = None

    def run(self):
        self.build_regression_model()
        start = time.time()
        self.fit()
        end = time.time()
        self.fitting_time = end - start
        self.predict()
        self.results = evaluation.evaluate_regressor(self.partitions, self.prediction, self.fitting_time)

    def run_classification(self):
        self.build_classification_model()
        start = time.time()
        self.fit()
        end = time.time()
        self.fitting_time = end - start
        self.predict()
        # get discrete values for prediction
        self.prediction = np.where(self.prediction < 0.5, 0., 1.)  # only for classification
        self.results = evaluation.evaluate_classifier(self.partitions, self.prediction, self.fitting_time)

    def build_regression_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=[95]),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        self.model = model

    def build_classification_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=[95]),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(23, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def fit(self):
        EPOCHS = 100
        # the patience parameter is the amount of epochs to check for improvement
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(self.partitions.x_train, self.partitions.y_train,
                       epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop], verbose=0)

    def predict(self):
        self.prediction = self.model.predict(self.partitions.x_test, batch_size=10, verbose=0)


class DummyRegressor:
    def __init__(self, partitions):
        self.partitions = partitions
        self.prediction = None
        self.results = None

    def run(self):
        self.predict()
        self.evaluate()

    def predict(self):
        # mean value of test as the random prediction
        self.prediction = np.repeat(np.mean(self.partitions.y_test), self.partitions.y_test.shape[0])

    def evaluate(self):
        self.results = evaluation.evaluate_regressor(self.partitions, self.prediction, 0)


class DummyClassifier:
    def __init__(self, partitions):
        self.partitions = partitions
        self.prediction = None
        self.results = None

    def run(self):
        self.predict()
        self.evaluate()

    def predict(self):
        # choose the value more spotted by column (0 or 1)
        df = pd.DataFrame(data=self.partitions.y_train)
        prediction = pd.DataFrame()
        n_rows = self.partitions.y_train.shape[0]
        n_columns = self.partitions.y_train.shape[1]
        for column in range(0, n_columns):
            ones = (df.iloc[:, column].values == 1).sum()
            greater = np.where(ones < n_rows/2, 0., 1.)
            prediction.insert(column, column, value=np.repeat(greater, self.partitions.x_test.shape[0]))

        self.prediction = prediction.to_numpy()

    def evaluate(self):
        self.results = evaluation.evaluate_classifier(self.partitions, self.prediction, 0)
