from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def load_extended_boston():
    boston = load_boston()
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

    return X, boston.target

X, y = load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)

print("train accuracy: {:.2f}".format(ridge.score(X_train, y_train)))
print("test  accuracy: {:.2f}".format(ridge.score(X_test, y_test)))
