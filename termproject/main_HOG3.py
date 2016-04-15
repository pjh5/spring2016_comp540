import utils
import numpy as np
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
from sklearn import cross_validation
import csv
import sys
from matplotlib import pyplot as plt

classes=["airplaine", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


ntrain = 50000

print ("Reading the training data...")
sys.stdout.flush()

TRAIN_DIR = "train/"

# use HOG as a list of features

train_imgs = utils.read_folder(TRAIN_DIR, 0, ntrain, flatten = False)
print ("\nDone!")
sys.stdout.flush()
print ("Getting HOG3 of the data...")
sys.stdout.flush()
X = utils.getHOG3(train_imgs, cpb=(1,1))
print ("\nDone!")
sys.stdout.flush()
X = np.insert(X, 0, 1.0, axis = 1)
theta = np.random.randn(X.shape[1], 10) * 0.0001
y = utils.read_labels('trainLabels.csv', 0, ntrain)
best_val = -1
best_softmax = None
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size = 0.1)

print "y_train.shape=", y_train.shape
print "y_val.shape=", y_val.shape
print "X_train.shape=", X_train.shape
print "X_val.shape=", X_val.shape
sys.stdout.flush()

# OVA isn't doing well (not better than Soft Max) but is too slow and is unable to 
# work with large amount of files (50 000 froze my laptop completely until reboot)
#classify = one_vs_allLogisticRegressor(np.arange(10))
#best_params, best_classifier = utils.getBestRegOVA(classify, X_train, y_train, X_val, y_val, regularization_strengths, pen='l1')

"""
#best LR = [15]
learning_rates = [15] # [11,12,13,14,15,16,17,18,19,20]
#Best RS = [1e-3]
regularization_strengths = [2e-3] #[1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3]
classify = SoftmaxClassifier()
# takes classifier, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths and optionally print_train and print_val
best_params, best_classifier = utils.getBestRegAndLearnSoftMax(classify, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths)
# end
"""
print "\nTraining the classifier..."
best_classifier = SoftmaxClassifier()
best_classifier.train(X_train, y_train, reg=1e1, learning_rate=15)
print np.mean(best_classifier.predict(X_val) == y_val)
print theta.shape
hogimg = best_classifier.theta[1025:1537,:].reshape(8,8,8,10)

utils.visualizeHOGTheta(hogimg)

print "\nMaking the final prediction..."
sys.stdout.flush()
labels = []
ids = []
batch = 50000
for j in range(3, 6):
	print "\nPart ", j + 1, " of 3"
	sys.stdout.flush()
	
	name = "batch_" + str(j) + ".npz"
	test_imgs = np.load(name)['arr_0']
	
	X_test = utils.getHOG3(test_imgs)
	X_test = np.insert(X_test, 0, 1.0, axis = 1)
	prediction = best_classifier.predict(X_test)
	for i in range(len(prediction)):
		labels.append(classes[prediction[i]])
		ids.append(j * batch + i + 1)

ids = np.array(ids).reshape((len(ids), 1))
labels = np.array(labels).reshape((len(labels), 1))

ids = np.array(ids).reshape(1,len(ids))
labels = np.array(labels).reshape(1,len(labels))
l = np.zeros(ids.size, dtype=[('var1', int), ('var2', 'S21')])
l['var1'] = ids
l['var2'] = labels

np.savetxt('testLabels2_part2.csv', l, fmt="%d,%s", delimiter=',')

print ("\nDone!")
