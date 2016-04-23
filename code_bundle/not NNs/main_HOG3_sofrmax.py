import utils
import numpy as np
from softmax import SoftmaxClassifier
from sklearn import cross_validation
import csv
import sys

classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

##!! THIS SCRIPT WITH ntrain = 50000 REQUIRES AROUND 10 GB RAM !!##


#changing this number allows to test the code with a smaller number of training images

##!! THIS SCRIPT WITH ntrain = 50000 REQUIRES AROUND 10 GB OF RAM !!##
ntrain = 50000

print ("Reading the training data...")
sys.stdout.flush()

TRAIN_DIR = "train/"
TEST_DIR = "test/"

# use HOG as a list of features
# reading in the data. This takes a while
train_imgs = utils.read_folder(TRAIN_DIR, 0, ntrain, flatten = False)
print ("\nDone!")
sys.stdout.flush()
print ("Getting HOG3 of the data...")
sys.stdout.flush()
# also takes a while
X = utils.getHOG3(train_imgs)
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

# this part was used to the best hyperparameters

#best LR = [15]
learning_rates = [15] # [11,12,13,14,15,16,17,18,19,20]
#Best RS = [1e-3]
regularization_strengths = [2e-3] #[1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3]
classify = SoftmaxClassifier()
# takes classifier, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths and optionally print_train and print_val
best_params, best_classifier = utils.getBestRegAndLearnSoftMax(classify, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths)
# end

#uncomment this to make prediction on the test data
"""
print "\nTraining the classifier..."
best_classifier = SoftmaxClassifier()
best_classifier.train(X_train, y_train, reg=1e-3, learning_rate=15)

print "\nMaking the final prediction..."
sys.stdout.flush()
labels = []
ids = []
batch = 50000
for j in range(0, 6):
	print "\nPart ", j + 1, " of 3"
	sys.stdout.flush()
	
	test_imgs = utils.read_folder(TEST_DIR, j * batch, (j + 1) * batch)
	
	X_test = utils.getHOG3(test_imgs)
	X_test = np.insert(X_test, 0, 1.0, axis = 1)
	prediction = best_classifier.predict(X_test)
	for i in range(len(prediction)):
		labels.append(classes[prediction[i]])
		ids.append(j * batch + i + 1)

ids = np.array(ids).reshape((len(ids), 1))
labels = np.array(labels).reshape((len(labels), 1))

l = np.concatenate((ids, labels), axis=1)

# write output file
out = open('testLabels.csv', 'w')
out.write('%s' % 'id,label')
out.write('\n')

for row in l:
	out.write('%s,' % row[0])
	out.write('%s' % row[1])
	out.write('\n')
out.close()
"""
print ("\nDone!")
