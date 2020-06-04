__author__ = '{Alfonso Aguado Bustillo}'

from sklearn import metrics
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def evaluate_regressor(partitions, prediction, fitting_time):
    mae = metrics.mean_absolute_error(partitions.y_test, prediction)
    mse = metrics.mean_squared_error(partitions.y_test, prediction)
    rmse = np.sqrt(metrics.mean_squared_error(partitions.y_test, prediction))
    spearman = spearmanr(partitions.y_test, prediction)
    train_test_size = str(len(partitions.x_train)) + "/" + str(len(partitions.x_test))
    results = {'train_test_size': train_test_size,
               'mae': mae,
               'mse': mse,
               'rmse': rmse,
               'spearman': spearman,
               'time': fitting_time}

    return results


def evaluate_classifier(partitions, prediction, fitting_time):
    # confusion matrices
    results = multilabel_confusion_matrix(partitions.y_test, prediction)

    total = prediction.shape[0]  # total number of sequences
    train_test_size = str(len(partitions.x_train)) + "/" + str(len(partitions.x_test))
    results = {'train_test_size': train_test_size,
               '1': str(round((results[0][0][0] + results[0][1][1]) / total * 100)) + '%',
               '2': str(round((results[1][0][0] + results[1][1][1]) / total * 100)) + '%',
               '3': str(round((results[2][0][0] + results[2][1][1]) / total * 100)) + '%',
               '4': str(round((results[3][0][0] + results[3][1][1]) / total * 100)) + '%',
               '5': str(round((results[4][0][0] + results[4][1][1]) / total * 100)) + '%',
               '6': str(round((results[5][0][0] + results[5][1][1]) / total * 100)) + '%',
               '7': str(round((results[6][0][0] + results[6][1][1]) / total * 100)) + '%',
               '8': str(round((results[7][0][0] + results[7][1][1]) / total * 100)) + '%',
               '9': str(round((results[8][0][0] + results[8][1][1]) / total * 100)) + '%',
               '10': str(round((results[9][0][0] + results[9][1][1]) / total * 100)) + '%',
               '11': str(round((results[10][0][0] + results[10][1][1]) / total * 100)) + '%',
               '12': str(round((results[11][0][0] + results[11][1][1]) / total * 100)) + '%',
               '13': str(round((results[12][0][0] + results[12][1][1]) / total * 100)) + '%',
               '14': str(round((results[13][0][0] + results[13][1][1]) / total * 100)) + '%',
               '15': str(round((results[14][0][0] + results[14][1][1]) / total * 100)) + '%',
               '16': str(round((results[15][0][0] + results[15][1][1]) / total * 100)) + '%',
               '17': str(round((results[16][0][0] + results[16][1][1]) / total * 100)) + '%',
               '18': str(round((results[17][0][0] + results[17][1][1]) / total * 100)) + '%',
               '19': str(round((results[18][0][0] + results[18][1][1]) / total * 100)) + '%',
               '20': str(round((results[19][0][0] + results[19][1][1]) / total * 100)) + '%',
               '21': str(round((results[20][0][0] + results[20][1][1]) / total * 100)) + '%',
               '22': str(round((results[21][0][0] + results[21][1][1]) / total * 100)) + '%',
               '23': str(round((results[22][0][0] + results[22][1][1]) / total * 100)) + '%',
               'time': round(fitting_time)}

    return results

