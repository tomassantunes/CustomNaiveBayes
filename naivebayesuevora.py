import pandas as pd
import numpy as np

def pre_processing(d):

    X = d.drop([d.columns[0]], axis = 1)
    y = d[d.columns[0]]

    return X, y

class NaiveBayesUevora:
    alpha = 0
    def __init__(self):

        self.features = []
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}
        
        self.train_size = 0
        self.num_feats = 0

        self.alpha = int(input("Defina o alpha a usar: "))
           
    def estimador(self, xi, N, d) -> float:
        return (xi + self.alpha) / (N + self.alpha*d)

    def fit(self, X, y):
        self.features = list(X.columns)
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}

        self.calc_class_prior()
        self.calc_likelihoods()
        self.calc_predictor_prior()

    def calc_class_prior(self):
        for i in np.unique(y):
            z = sum(y == i)
            self.class_priors[i] = z / self.train_size

    def calc_likelihoods(self):
        for feature in self.features:
            d = len(np.unique(X[feature]))
            
            for j in np.unique(X[feature]):
                xi = 0

                for k in X[feature]:
                    if(j == k):
                        xi += 1

                for l in np.unique(y):
                    N = 0
                
                    for n in y:
                        if(n == l):
                            N += 1

                    self.likelihoods[feature][str(j) + '_' + str(l)] = self.estimador(xi, N, d)

    def calc_predictor_prior(self):
        for feature in self.features:
            priors = np.unique(X[feature])
            
            for i in priors:
                count = 0

                for j in X[feature]:
                    if j == i:
                        count += 1
                
                self.pred_priors[feature][i] = count / self.train_size

    # def predict(self, X, y):

    # def accuracy_score(self, X, y):

    # def precision_score(self, X, y):

nbue = NaiveBayesUevora()

data_train = pd.read_csv("breast.cancer-train.csv")
data_test = pd.read_csv("breast-cancer-test.csv")

X, y = pre_processing(data_train)

nbue.fit(X, y)

print(nbue.class_priors)
print(nbue.likelihoods)
print(nbue.pred_priors)

# https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9
