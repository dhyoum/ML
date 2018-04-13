from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
print("feature max :\n{}".format(cancer.data.max(axis=0)))
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
        stratify=cancer.target, random_state=0)

for layer in range(1,2):
    mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(100*layer,), max_iter=100)
    mlp.fit(X_train, y_train)

    print("train accuracy: {:.3f}".format(mlp.score(X_train, y_train)))
    print("test  accuracy: {:.3f}".format(mlp.score(X_test, y_test)))
