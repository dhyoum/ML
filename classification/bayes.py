import numpy as np

X = np.array([[0,1,0,1],
              [1,0,1,1],
              [0,0,0,1],
              [1,0,1,0]])
y = np.array([0,1,0,1])

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()

model.fit(X, y)

print(model.score(X,y))
print(model.predict(X))
