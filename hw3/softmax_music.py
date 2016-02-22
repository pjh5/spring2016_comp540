import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import music_utils
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report
import time

# TODO: Get the music dataset (CEFS representation) [use code from Hw2]
MUSIC_DIR = "../music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
X,y = music_utils.read_ceps(genres,MUSIC_DIR)
#X,y = music_utils.read_fft(genres,MUSIC_DIR)

# TODO: Split into train, validation and test sets 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_train, y_train, test_size=0.1)


# TODO: Use the validation set to tune hyperparameters for softmax classifier
# choose learning rate and regularization strength (use the code from softmax_hw.py)

results = {}
best_val = -1
best_softmax = SoftmaxClassifier()
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
regularization_strengths = [5e4, 1e5, 5e5, 1e8]

classifier = SoftmaxClassifier()
accuracy = lambda x,y: np.mean(classifier.predict(x) == y)
for lr in learning_rates:
  for reg in regularization_strengths:
    classifier.train(X_train, y_train, learning_rate=lr, reg=reg)
    cvaccuracy = accuracy(X_val, y_val)
    results[(lr, reg)] = (accuracy(X_train, y_train), cvaccuracy)
    
    if (cvaccuracy > best_val):
      best_softmax.theta = classifier.theta
      best_val = cvaccuracy


# TODO: Evaluate best softmax classifier on set aside test set (use the code from softmax_hw.py)
# Print out results.
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate the best softmax classifier on test set

y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )


# TODO: Compare performance against OVA classifier of Homework 2 with the same
# train, validation and test sets (use sklearn's 

ypred = ova_logreg.predict(X_test)
print confusion_matrix(y_test,ypred)classifier evaluation metrics)

ova_logreg = one_vs_allLogisticRegressor(np.arange(10))
ova_logreg.train(X_train,y_train,1/regularization_strengths[best_reg[0]],'l2')
ypred = ova_logreg.predict(X_test)
print ('OVA:')
print ('OVA final test set accuracy:', np.mean(ypred==y_test))
print (confusion_matrix(y_test,ypred))
print ('Soft max:')
print ('Softmax final test set accuracy: %f' % (test_accuracy, ))
print (confusion_matrix(y_test,y_test_pred))
