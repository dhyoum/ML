from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

dataset = pd.read_csv('../data/diabetes.csv', names=[1,2,3,4,5,6,7,8,9], header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum())
print(dataset.describe())
print(dataset[9].head())

X = dataset[[1,2,3,4,5,6,7,8]]
y = dataset[9]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state=107)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.predict(X_test[0:10]))
predictions = lr.predict(X_test)

score = lr.score(X_test, y_test)
print(score)
