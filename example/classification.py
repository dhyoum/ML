from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv("diabetes.csv", names=[1,2,3,4,5,6,7,8,9], header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum())
print(dataset.describe())
print(dataset[9].head())

X = dataset[[1,2,3,4,5,6,7,8]]
y = dataset[9]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state=107)

lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)

print(" LR : %s" % lr.score(X_test, y_test))
print("KNN : %s" % knn.score(X_test, y_test))
print("DT  : %s" % dt.score(X_test, y_test))
print("RF  : %s" % rf.score(X_test, y_test))
print("SVM : %s" % svm.score(X_test, y_test))
print("MLP : %s" % mlp.score(X_test, y_test))
