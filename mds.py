# dataY.txtをMDSで2次元平面上にプロットする。
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

mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
pos = mds.fit_transform(data)

plt.figure(figsize=(30, 30))
for index in range(9):
	plt.subplot(3, 3, index + 1)
	num_str = 'th'
	if index == 1:
		num_str = 'st'
	elif index == 2:
		num_str = 'nd'
	elif index == 3:
		num_str = 'rd'
		
	plt.title('Labeled by ' + str(index) + num_str + ' component of PM parameter')
	plt.scatter(pos[label[:,index] == -1, 0], pos[label[:,index] == -1, 1], s=50, c="#000000")
	plt.scatter(pos[label[:,index] == 0, 0], pos[label[:,index] == 0, 1], s=50, c="#888888")
	plt.scatter(pos[label[:,index] == 1, 0], pos[label[:,index] == 1, 1], s=50, c="#ffffff")
	plt.legend(('-1','0','1'), loc='best')

plt.savefig('mds.png', dpi=200)
plt.show()
