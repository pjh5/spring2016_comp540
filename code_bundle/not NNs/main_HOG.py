import utils
import numpy as np
from softmax import SoftmaxClassifier
from sklearn import cross_validation
import csv
import sys

classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#changing this number allows to test the code with a smaller number of training images
ntrain = 50000

print ("Reading the training data...")
sys.stdout.flush()

TRAIN_DIR = "train/"
TEST_DIR = "test/"

# use HOG as a list of features
# reading in the data. This takes a while
train_imgs = utils.read_folder(TRAIN_DIR, 0, ntrain, flatten = True)
print ("\nDone!")
sys.stdout.flush()
print ("Getting HOG of the data...")
sys.stdout.flush()
# also takes a while
X = utils.getHOG(train_imgs, vis=False)
print ("\nDone!")
sys.stdout.flush()
X = np.insert(X, 0, 1.0, axis = 1)
theta = np.random.randn(X.shape[1], 10) * 0.0001
y = utils.read_labels('trainLabels.csv', 0, ntrain)

best_val = -1
best_softmax = None

#these two values below are the best
learning_rates = [3]
regularization_strengths = [1e-1]

X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size = 0.1)

print "y_train.shape=", y_train.shape
print "y_val.shape=", y_val.shape
print "X_train.shape=", X_train.shape
print "X_val.shape=", X_val.shape
sys.stdout.flush()

classify = SoftmaxClassifier()
# takes classifier, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths and optionally print_train and print_val
best_params, best_classifier = utils.getBestRegAndLearnSoftMax(classify, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths)
# end

#two lines below will create visualization of theta, if uncommented
#hogimg = classify.theta[1:,:].reshape(8,8,8,10)
#utils.visualizeHOGTheta(hogimg)

#uncomment this to make prediction on the test data
"""
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
for row in l:
    for column in row:
        out.write('%s,' % column)
    out.write('\n')
out.close()

print ("Done!")
"""