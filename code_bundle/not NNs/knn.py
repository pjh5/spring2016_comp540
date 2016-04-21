import numpy as np
import sys

class KNNClassifier:

	def __init__(self, n = 10):
		self.X_tr = None
		self.y_tr = None
		self.n = n

	def train(self, X, y):
		self.X_tr = X
		self.y_tr = y
		return 0

	def predict(self, X):
		print "K =", self.n
		m = X.shape[0]
		y_pred = np.zeros(m)
		for i in range(m):
			dist = np.sum(np.abs(self.X_tr - X[i,:]), axis = 1)
			if self.n == 1:
				min = np.argmin(dist)
				y_pred[i] = self.y_tr[min]
			else:
				votes = dist.argsort()[:self.n]
				v_cats = self.y_tr[votes]
				y_pred[i] = np.argmax(np.bincount(v_cats))
			sys.stdout.write("\rIteration {0}/{1}".format(i, m))
			sys.stdout.flush()
		print "\n"
		return y_pred