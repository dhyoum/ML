from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

print("Image Data Share", digits.data.shape)
print("Label Data Share", digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title("Training: %i\n" % label, fontsize=20)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0, test_size=.25)

from sklearn.linear_model import LogisticRegression


lr = LogisticRegression()
lr.fit(X_train, y_train)

print("Logistic Regression Train score : %s\n" % lr.score(X_train, y_train))
print("Logistic Regression Test  score : %s\n" % lr.score(X_test, y_test))

print("Logistic Regression Predict %s\n" % lr.predict(X_test[0].reshape(1,-1)))
print("Logistic Regression Answer  %s\n" % y_test[0])
