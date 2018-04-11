from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
        stratify=cancer.target, random_state=42)

for depth in range(1,5):
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=depth)
    gbrt.fit(X_train, y_train)
    print("depth {} train accuracy: {:.3f}".format(depth, gbrt.score(X_train, y_train)))
    print("depth {} test  accuracy: {:.3f}".format(depth, gbrt.score(X_test, y_test)))
