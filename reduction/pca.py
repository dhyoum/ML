from sklearn.preprocessing.data import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print("original {}, reduction {}".format(X_scaled.shape, X_pca.shape))

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
plt.legend(["malignancy(cancer)", "benign"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.draw()

print("PCA PC shape:{}".format(pca.components_.shape))
print("PCA PC {}".format(pca.components_))
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0,1], ["first principal component", "second principal component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
        cancer.feature_names, rotation=60, ha='left')

plt.xlabel('feature')
plt.ylabel('principal component')
plt.show()
