from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
        stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("train accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("test  accuracy: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["cancer", "normal"],
        feature_names=cancer.feature_names, impurity=False, filled=True)
from graphviz import Source
Source.from_file("tree.dot").view()


