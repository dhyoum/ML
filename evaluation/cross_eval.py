from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

iris = load_iris()
logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("iris.feature : {}".format(X_train.shape))

logreg = LogisticRegression().fit(X_train, y_train)
print("test accuracy: {:.2f}, coef:{}".format(logreg.score(X_test, y_test), logreg.coef_))

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("cross validation: {}".format(scores))

logreg = LogisticRegression()
kfold = KFold(n_splits=5)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("cross validation: {}".format(scores))
