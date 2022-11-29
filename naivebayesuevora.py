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
        self.test_size = 0
        self.num_feats = 0

        self.y_train = []

        self.alpha = int(input("Defina o alpha a usar: "))
           
    def estimador(self, xi, N, d) -> float:
        return (xi + self.alpha) / (N + (self.alpha*d))

    def fit(self, X, y):
        self.features = list(X.columns)
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]
        self.y_train = y

        for feature in self.features:
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}

        self.calc_class_prior()
        self.calc_likelihoods()
        self.calc_predictor_prior()

    def calc_class_prior(self): # apenas class principal (Class)
        for i in np.unique(y):
            z = sum(y == i)
            self.class_priors[i] = z / self.train_size

    def calc_likelihoods(self): # probabilidade de class principal com restos das classes
        for feature in self.features:
            d = len(np.unique(X[feature]))
            
            for j in np.unique(X[feature]):
                
                for l in np.unique(y):
                    N = 0
                    xi = 0

                    for k in range(self.train_size):
                        if(X[feature][k] == j and y[k] == l):
                            xi += 1

                    for n in y:
                        if(n == l):
                            N += 1

                    self.likelihoods[feature][str(j) + '_' + str(l)] = self.estimador(xi, N, d)

    def calc_predictor_prior(self): # probabilidade das outras classes existirem
        for feature in self.features:
            priors = np.unique(X[feature])
            
            for i in priors:
                count = 0
    
                for j in X[feature]:
                    if(j == i):
                        count += 1
                
                self.pred_priors[feature][i] = count / self.train_size

    def check_attribute(self, X, attr, feature, class_name=""):
        if class_name != "":
            x = f"{attr}_{class_name}"
            if x not in self.likelihoods[feature]:
                xi = 0
                for k in X[feature]:
                    if(k == attr):
                        xi += 1

                return self.estimador(xi, 1, len(np.unique(X[feature])))
            return self.likelihoods[feature][x]

        if attr not in self.pred_priors[feature]:
            count = 0

            for i in X[feature]:

                if (i == attr):
                    count += 1

            return count / self.test_size
        return self.pred_priors[feature][attr]

    def predict(self, X):
        y = np.unique(self.y_train)
        self.test_size = X.shape[0]
        results = []

        for j in range(self.test_size):
            probs_outcome = {}

            for i in y:
                tmpTop = 1
                tmpBot = 1

                for feature in self.features:
                    tmpTop *= self.check_attribute(X, X[feature][j], feature, i)
                    # tmpBot *= self.check_attribute(X, X[feature][j], feature)
                    tmpBot *= 1

                probs_outcome[i] = (tmpTop * self.class_priors[i]) / tmpBot

            result = max(probs_outcome, key = lambda x: probs_outcome[x])
            print(probs_outcome)
            results.append(result)
        
        return results

    # def accuracy_score(self, X, y):

    # def precision_score(self, X, y):

nbue = NaiveBayesUevora()

data_train = pd.read_csv("breast.cancer-train.csv")
data_test = pd.read_csv("breast-cancer-test.csv")

X, y = pre_processing(data_train)

nbue.fit(X, y)

X_test, y_test = pre_processing(data_test)

print(nbue.class_priors)
print(nbue.likelihoods)
print(nbue.pred_priors)

nbue.predict(X_test)

# https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9
