# dataY.txtをLTSAで2次元平面上にプロットする。
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

lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=10, eigen_solver='auto', method='modified')
pos = lle.fit_transform(data)


fig = plt.figure(1)

plt.scatter(pos[label[:,1] == -1, 0], pos[label[:,1] == -1, 1], s=60, c="#000000")
plt.scatter(pos[label[:,1] == 0, 0], pos[label[:,1] == 0, 1], s=60, c="#888888")
plt.scatter(pos[label[:,1] == 1, 0], pos[label[:,1] == 1, 1], s=60, c="#ffffff")
plt.legend(('-1','0','1'), loc='best')

plt.show()