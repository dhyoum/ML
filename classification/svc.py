from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
        stratify=cancer.target, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)

print("train accuracy: {:.3f}".format(svc.score(X_train, y_train)))
print("test  accuracy: {:.3f}".format(svc.score(X_test, y_test)))
