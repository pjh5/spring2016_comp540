import numpy as np
from sklearn import cross_validation
from sklearn.svm import LinearSVC
import csv, sys, utils

classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

##!! THIS SCRIPT WITH ntrain = 50000 REQUIRES AROUND 12 GB RAM !!##

#changing this number allows to test the code with a smaller number of training images
##!! THIS SCRIPT WITH ntrain = 50000 REQUIRES AROUND 12 GB RAM !!##
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

#best LR = [15]
#learning_rates = [15] # [11,12,13,14,15,16,17,18,19,20]
#Best RS = [1e-3]
#regularization_strengths = [2e-3] #[1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3]
# best C = .36

# best parameter value
C = [.36]
# this will train SVM and print out validation accuracy
print "C -> val_acc"
print "------------"
for i in range(len(C)):
	classify = LinearSVC(C=C[i])
	classify.fit(X_train, y_train)
	pred = classify.predict(X_val)
	print C[i], "->", np.average(pred == y_val)


best_classifier = LinearSVC(C=0.36)
best_classifier.fit(X_train, y_train)


#uncomment this to make prediction on the test data
"""
print "\nMaking the final prediction..."
sys.stdout.flush()
labels = []
ids = []
batch = 50000
for j in range(0, 6):
	print "\nPart ", j + 1, " of 2"
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

ids = np.array(ids).reshape(1,len(ids))
labels = np.array(labels).reshape(1,len(labels))
l = np.zeros(ids.size, dtype=[('var1', int), ('var2', 'S21')])
l['var1'] = ids
l['var2'] = labels

np.savetxt('testLabelsSVM_part3.csv', l, fmt="%d,%s", delimiter=',')
"""
print ("\nDone!")
