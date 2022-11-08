import pandas as pd
import numpy as np

class NaiveBayesUevora:
    alpha = 0
    def __init__(self):
        alpha = float(input("Defina o alpha a usar: "))
           
    def estimador(self, xi, N, d) -> float:
        return (xi + self.alpha) / (N + self.alpha*d)

    def fit(self, x, y):
        self.num_classes = len(np.unique(y))

        self.classes_mean = {}
        self.classes_prior = {}
        self.classes_varience = {}

        for classes in range(self.num_classes):
            x_c = x[y == classes]

            self.classes_mean[str(classes)] = np.mean(x_c, axis=0)
            self.classes_prior[str(classes)] = x_c.shape[0] / x.shape[0]
            self.classes_varience[str(classes)] = np.var(x_c, axis=0)

    # def predict(self, x, y):

    # def accuracy_score(self, x, y):

    # def precision_score(self, x, y):

nbue = NaiveBayesUevora()

data_test = pd.read_csv("breast-cancer-test.csv")
data_train = pd.read_csv("breast.cancer-train.csv")

X_test = data_test
y_test = data_test['Class']

X_train = data_train
y_train = data_train['Class']

nbue.fit(X_train, y_train)