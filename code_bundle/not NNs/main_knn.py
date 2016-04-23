import utils
from knn import KNNClassifier
import numpy as np
from sklearn import cross_validation
import csv
import sys
from matplotlib import pyplot as plt

classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#changing this number allows to test the code with a smaller number of training images
ntrain = 5000

print ("Reading the training data...")
sys.stdout.flush()

TRAIN_DIR = "train/"
TEST_DIR = "test/"

# use simple self-written KNN
train_imgs = utils.read_folder(TRAIN_DIR, 0, ntrain, flatten=False)
print ("\nDone!")
sys.stdout.flush()
X = train_imgs
X = X.reshape((ntrain, -1))
#X = np.insert(X, 0, 1.0, axis = 1)

y = utils.read_labels('trainLabels.csv', 0, ntrain)



X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size = 0.1)
nns = [1]
#utils.getBestK(X_train, y_train, X_val, y_val, nns)

knn = KNNClassifier(nns[0])
knn.train(X_train, y_train)
print "X_val shape: ", X_val.shape
print "y_val shape: ", y_val.shape
pred = knn.predict(X_val)
print "Accuracy: ", np.mean(pred == y_val)
#uncomment this to visualize knn prediction - 10 examples from each class
"""
examples = np.zeros((10,10,32,32,3))
for i in range(10):
	examples[i] = ((X_val[pred==i])[0:10]).reshape(10,32,32,3)
num_classes = len(classes)
nexamples = 10
for y, cls in enumerate(classes):
	idxs = np.arange(nexamples)
	for i, idx in enumerate(idxs):
		plt_idx = i * num_classes + y + 1
		plt.subplot(nexamples, num_classes, plt_idx)
		plt.imshow(examples[y, i].astype('uint8'), cmap=plt.cm.gray)
		plt.axis('off')
		if i == 0:
			plt.title(cls)
plt.savefig('1nn.pdf')
plt.close()
"""
#uncomment this to make prediction on the test data
"""
labels = ['label']
ids = ['id']
print ("\nMaking the final prediction...")
sys.stdout.flush()

batch = 50000
for j in range(0, 6):
	print ("Part ", j + 1, " of 6")
	sys.stdout.flush()
	
	test_imgs = utils.read_folder(TEST_DIR, j * batch, (j + 1) * batch)
	X_test = utils.getHOG(test_imgs)
	X_test = np.insert(X_test, 0, 1.0, axis = 1)
	prediction = best_classifier.predict(X_test)
	for i in range(len(prediction)):
		labels.append(classes[prediction[i]])
		ids.append(j * batch + i + 1)

ids = np.array(ids).reshape((len(ids), 1))
labels = np.array(labels).reshape((len(labels), 1))

l = np.concatenate((ids, labels), axis=1)

out = open('testLabels.csv', 'w')
out.write('%s' % 'id,label')
out.write('\n')

for row in l:
	out.write('%s,' % row[0])
	out.write('%s' % row[1])
	out.write('\n')
out.close()
"""
print ("Done!")
