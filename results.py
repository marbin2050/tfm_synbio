__author__ = '{Alfonso Aguado Bustillo}'

from prettytable import PrettyTable


def show_regression_results(regression_results):

    regression_table = PrettyTable()
    regression_table.field_names = ["algorithm", "train/test size", "mae", "mse", "rmse", "spearman", "time"]

    for result in regression_results:
        regression_table.add_row([result,
                                  regression_results.get(result)['train_test_size'],
                                  round(regression_results.get(result)['mae'], 2),
                                  round(regression_results.get(result)['mse'], 2),
                                  round(regression_results.get(result)['rmse'], 2),
                                  "corr=" + str(round(regression_results.get(result)['spearman'].correlation, 2))
                                  + ", " + "p=" + str(round(regression_results.get(result)['spearman'].pvalue, 2)),
                                  round(regression_results.get(result)['time'], 2)])
    regression_table.sortby = 'mae'

    print(regression_table)


def show_classification_results(classification_results):
    classification_table = PrettyTable()
    # classification_table.field_names = ["algorithm", "train/test", "1", "2", "3", "4", "5",
    #                                     "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
    #                                     "16", "17", "18", "19", "20", "21", "22", "23", "precision", "recall",
    #                                     "f1", "time"]

    classification_table.field_names = ["algorithm", "1", "2", "3", "4", "5",
                                        "6", "7", "8", "9", "10", "11", "12", "precision", "recall",
                                        "f1", "time"]

    for result in classification_results:
        classification_table.add_row([result,
                                      # classification_results.get(result)['train_test_size'],
                                      classification_results.get(result)['1'],
                                      classification_results.get(result)['2'],
                                      classification_results.get(result)['3'],
                                      classification_results.get(result)['4'],
                                      classification_results.get(result)['5'],
                                      classification_results.get(result)['6'],
                                      classification_results.get(result)['7'],
                                      classification_results.get(result)['8'],
                                      classification_results.get(result)['9'],
                                      classification_results.get(result)['10'],
                                      classification_results.get(result)['11'],
                                      classification_results.get(result)['12'],
                                      # classification_results.get(result)['13'],
                                      # classification_results.get(result)['14'],
                                      # classification_results.get(result)['15'],
                                      # classification_results.get(result)['16'],
                                      # classification_results.get(result)['17'],
                                      # classification_results.get(result)['18'],
                                      # classification_results.get(result)['19'],
                                      # classification_results.get(result)['20'],
                                      # classification_results.get(result)['21'],
                                      # classification_results.get(result)['22'],
                                      # classification_results.get(result)['23'],
                                      classification_results.get(result)['precision score'],
                                      classification_results.get(result)['recall score'],
                                      classification_results.get(result)['f1 score'],
                                      classification_results.get(result)['time']])

    classification_table.sortby = 'f1'
    classification_table.reversesort = True

    print(classification_table)
