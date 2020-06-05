from sklearn.model_selection import train_test_split


class Partitions:
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def create_data_partitions(self, x, y, test_size=0.2):
        # split in train and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=0)