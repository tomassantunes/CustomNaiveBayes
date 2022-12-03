import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

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

        self.calc_class_prior(X, y)
        self.calc_likelihoods(X, y)
        self.calc_predictor_prior(X, y)

    def calc_class_prior(self, X, y): # apenas class principal (Class)
        for i in np.unique(y):
            z = sum(y == i)
            self.class_priors[i] = z / self.train_size

    def calc_likelihoods(self, X, y): # probabilidade de class principal com restos das classes
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

    def calc_predictor_prior(self, X, y): # probabilidade das outras classes existirem
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
                tmp = 1

                for feature in self.features:
                    # X[feature].keys()[j] -> ir buscar as keys do split
                    tmp *= self.check_attribute(X, X[feature][X[feature].keys()[j]], feature, i)

                probs_outcome[i] = (tmp * self.class_priors[i])

            result = max(probs_outcome, key = lambda x: probs_outcome[x])
            results.append(result)
        
        return results

    def accuracy_score(self, Y_pred, Y_test):
        return float(sum(Y_pred==Y_test))/float(len(Y_test))

    #precision(for each)= truePositives / truePositives + FalsePositives
    def precision_score(self, y_test, prediction):
        confusionMatrix = confusion_matrix(y_test, prediction)
        precisions=[]
        numberOfClasses = len(confusionMatrix[0])
        for i in range(numberOfClasses):
            numberOfFakePositives=0
            for j in range(numberOfClasses):
                if j != i:
                    numberOfFakePositives += confusionMatrix[j][i]
            precision= float(confusionMatrix[i][i]/ (numberOfFakePositives + confusionMatrix[i][i]))
            precisions.append(precision)

        return float(sum(precisions)/ len(precisions))