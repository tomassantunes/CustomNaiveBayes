import naivebayesuevora
import pandas as pd
from sklearn.model_selection import train_test_split

def pre_processing(d):

    X = d.drop([d.columns[0]], axis = 1)
    y = d[d.columns[0]]

    return X, y

# breast-cancer
nbue = naivebayesuevora.NaiveBayesUevora()

data_train = pd.read_csv("breast.cancer-train.csv")
data_test = pd.read_csv("breast-cancer-test.csv")

X, y = pre_processing(data_train)

nbue.fit(X, y)

X_test, y_test = pre_processing(data_test)

print(nbue.class_priors)
print(nbue.likelihoods)
print(nbue.pred_priors)

prediction = nbue.predict(X_test)
print(prediction)
print(nbue.accuracy_score(prediction,y_test))

print(nbue.precision_score(y_test, prediction))

# breast-cancer-2
nbue2 = naivebayesuevora.NaiveBayesUevora()

data_train2 = pd.read_csv("breast-cancer-train2.csv")
data_test2 = pd.read_csv("breast-cancer-test2.csv")

X2, y2 = pre_processing(data_train2)

nbue2.fit(X2, y2)

X_test2, y_test2 = pre_processing(data_test2)

print(nbue2.class_priors)
print(nbue2.likelihoods)
print(nbue2.pred_priors)

prediction2 = nbue2.predict(X_test2)
print(prediction2)
print(nbue2.accuracy_score(prediction2, y_test2))

print(nbue2.precision_score(y_test2, prediction2))

# weather-nominal
nbue3 = naivebayesuevora.NaiveBayesUevora()

weather = pd.read_csv("weather-nominal.csv")

X3 = weather.drop([weather.columns[-1]], axis = 1)
y3 = weather[weather.columns[-1]]

nbue3.fit(X3, y3)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3)

print(nbue3.class_priors)
print(nbue3.likelihoods)
print(nbue3.pred_priors)

prediction3 = nbue3.predict(X_test3)
print(prediction3)
print(nbue3.accuracy_score(prediction3, y_test3))

print(nbue3.precision_score(y_test3, prediction3))