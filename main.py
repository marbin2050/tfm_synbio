__author__ = '{Alfonso Aguado Bustillo}'

import partitions
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier, XGBRFRegressor
import preprocessing
import learning_model
import pandas as pd
from numpy import array
import numpy as np
from results import show_regression_results, show_classification_results

# COMMAND LINE ARGUMENTS
data = "input_files/dataset_C_10_LINEAR_NOISE90_cells.csv"


def predict_substitution_frequency(partitions_reg):

    regression_results = {}

    # DUMMY ALGORITHM PREDICTION
    random_reg = learning_model.DummyRegressor(partitions_reg)
    random_reg.run()
    regression_results['Dummy Regression'] = random_reg.results

    # MULTIPLE LINEAR REGRESSION
    estimator = LinearRegression()
    mlr = learning_model.Regressor(partitions_reg, estimator)
    mlr.run()
    regression_results['Linear Regression'] = mlr.results

    # SVM
    estimator = LinearSVR()
    svmr = learning_model.Regressor(partitions_reg, estimator)
    svmr.run()
    regression_results['SVM'] = svmr.results

    # DECISION TREE
    estimator = DecisionTreeRegressor()
    dtr = learning_model.Regressor(partitions_reg, estimator)
    dtr.run()
    regression_results['Decision Tree'] = dtr.results

    # RANDOM FOREST
    estimator = RandomForestRegressor()
    rfr = learning_model.Regressor(partitions_reg, estimator)
    rfr.run()
    regression_results['Random Forest'] = rfr.results

    # XGBOOST (regression with squared loss)
    estimator = XGBRegressor()
    xgbr = learning_model.Regressor(partitions_reg, estimator)
    xgbr.run()
    regression_results['XGBoost'] = xgbr.results

    # XGBOOST (random forest)
    estimator = XGBRFRegressor()
    xgbrf = learning_model.Regressor(partitions_reg, estimator)
    xgbrf.run()
    regression_results['XGBoostRF'] = xgbrf.results

    # MULTILAYER PERCEPTRON
    mlp = learning_model.MultilayerPerceptron(partitions_reg)
    mlp.run()
    regression_results['M. Perceptron'] = mlp.results

    # CONVOLUTIONAL NEURAL NETWORK
    cnn = learning_model.ConvolutionalNeuralNetwork(partitions_reg)
    cnn.run_regression()
    regression_results['CNN'] = cnn.results

    return regression_results


def predict_output_sequence(partitions_clf):

    classification_results = {}

    # DUMMY ALGORITHM PREDICTION
    dummy_clf = learning_model.DummyClassifier(partitions_clf)
    dummy_clf.run()
    classification_results['Dummy'] = dummy_clf.results

    # DECISION TREE
    estimator = DecisionTreeClassifier()
    dtc = learning_model.Classifier(partitions_clf, estimator)
    dtc.run()
    classification_results['Decision Tree'] = dtc.results

    # RANDOM FOREST
    estimator = RandomForestClassifier()
    rfc = learning_model.Classifier(partitions_clf, estimator)
    rfc.run()
    classification_results['Random Forest'] = rfc.results

    # LOGISTIC REGRESSION
    estimator = OneVsRestClassifier(LogisticRegression(max_iter=100000))
    lgc = learning_model.Classifier(partitions_clf, estimator)
    lgc.run()
    classification_results['Logistic'] = lgc.results

    # SVM
    estimator = OneVsRestClassifier(LinearSVC())
    svmc = learning_model.Classifier(partitions_clf, estimator)
    svmc.run()
    classification_results['SVM'] = svmc.results

    # XGBOOST
    estimator = OneVsRestClassifier(XGBClassifier())
    xgbc = learning_model.Classifier(partitions_clf, estimator)
    xgbc.run()
    classification_results['XGBoost'] = xgbc.results

    # MULTILAYER PERCEPTRON
    mlp = learning_model.MultilayerPerceptron(partitions_clf)
    mlp.run_classification()
    classification_results['M. Perceptron'] = mlp.results

    # CONVOLUTIONAL NEURAL NETWORK
    cnn = learning_model.ConvolutionalNeuralNetwork(partitions_clf)
    cnn.run_classification()
    classification_results['CNN'] = cnn.results

    return classification_results


def main():

    # STEP 0: GET DNA DATA
    df = pd.read_csv(data, header=0)

    # STEP 1: PREPARE INPUT AND OUTPUT DATA for REGRESSION AND CLASSIFICATION
    x = df.loc[:, 'gRNA']  # input values
    y_reg = df.loc[:, 'substitution_frequency']  # output values for regression
    y_reg = array(y_reg) * 100  # array type and scale to make it more interpretable
    y_clf = df.loc[:, 'sub_seq_edited']  # output values for classification

    # STEP 2: ONE-HOT ENCODING DNA DATA (nucleotide A)
    x1 = preprocessing.one_hot_encoding(x)  # one-hot encoding the input values (x)
    x2 = df[['GC_content_norm', 'hybridization_norm', 'score_norm']].to_numpy()
    x = np.concatenate((x1, x2), axis=1)
    nucleotide = "T"
    y_clf = preprocessing.binary_encoding(y_clf, nucleotide)  # encoding the output values for classification

    # STEP 3: SPLIT DATA INTO TRAIN, VALIDATION and TEST for REGRESSION
    partitions_reg = partitions.Partitions()
    partitions_reg.create_data_partitions(x, y_reg)

    # # STEP 4: SPLIT DATA INTO TRAIN, VALIDATION and TEST for CLASSIFICATION
    partitions_clf = partitions.Partitions()
    partitions_clf.create_data_partitions(x, y_clf)

    # STEP 5: PREDICTING THE SUBSTITUTION FREQUENCY (REGRESSION PROBLEM)
    # regression_results = predict_substitution_frequency(partitions_reg)
    # show_regression_results(regression_results)

    # STEP 6: PREDICTING THE SEQUENCE OUTPUT (CLASSIFICATION PROBLEM: ABE)
    classification_results = predict_output_sequence(partitions_clf)
    show_classification_results(classification_results)

    # TODO: Include total time in tables and, perhaps, calculate time including also predict
    # TODO: Automatize the evaluation and the results for the classification problem
    # TODO: Join MLP and CNN
    # TODO: Check the logistic regression warnings
    # TODO: Show a message at the beginning of each algorithm to help knowing where we are while executing
    # TODO: Introduce multi inputs
    # TODO: Add Learning and Validation curves.
    # TODO: Load hyperparameters instead of passing them as parameters
    # TODO: Comment code
    # TODO: Calculate fitting_time in neural networks by using callbacks


main()
