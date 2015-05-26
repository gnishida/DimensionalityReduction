# dataY.txtを2次元平面上にプロットする。
# 同時に、dataX.txt (9次元)をラベルとして表示する。

print(__doc__)
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


seed = np.random.RandomState(seed=3)

# read data
data = []
f = open('smallY.txt', 'r')
for line in f:
	rec = []
	for v in line.split(' '):
		rec.append(float(v))
	data.append(rec)
data = np.array(data);
f.close()

# read label
label = []
f = open('smallX.txt', 'r')
for line in f:
	rec = []
	for v in line.split(' '):
		rec.append(float(v))
	label.append(rec)
label = np.array(label);
f.close()

# Center the data
data -= data.mean()

plt.figure(figsize=(20, 4))

# PM parameterの、この番目の要素に基づいてラベルする
index = 8

# MDS
mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
pos = mds.fit_transform(data)
plt.subplot(1, 6, 1)
plt.title('MDS')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

# Isomap
pos = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(data)
plt.subplot(1, 6, 2)
plt.title('Isomap')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

# Spectral embedding
se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
pos = se.fit_transform(data)
plt.subplot(1, 6, 3)
plt.title('Spectral embedding')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

# t-SNE
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
pos = tsne.fit_transform(data)
plt.subplot(1, 6, 4)
plt.title('t-SNE')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

# LLE
lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=10, eigen_solver='auto', method='standard')
pos = lle.fit_transform(data)
plt.subplot(1, 6, 5)
plt.title('Locally Linear Embedding')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

# modified LLE
lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=10, eigen_solver='auto', method='modified')
pos = lle.fit_transform(data)
plt.subplot(1, 6, 6)
plt.title('Modified LLE')
plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")

#plt.savefig('se.png', dpi=200)
plt.show()