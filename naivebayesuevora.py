import pandas as pd

class NaiveBayesUevora:
    alpha = 0
    def __init__(self):
        alpha = float(input("Defina o alpha a usar: "))
           
    def estimador(self, xi, N, d) -> float:
        return (xi + self.alpha) / (N + self.alpha*d)

    def fit(self, x, y):
        print(x)
        print(y)

    # def predict(self, x, y):

    # def accuracy_score(self, x, y):

    # def precision_score(self, x, y):

nbue = NaiveBayesUevora()

data_test = pd.read_csv("breast-cancer-test.csv")
data_train = pd.read_csv("breast.cancer-train.csv")

nbue.fit(data_test, data_test['Class'])