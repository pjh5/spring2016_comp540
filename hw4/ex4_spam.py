from sklearn import preprocessing, metrics
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1

test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()

#############################################################################
# your code for setting up the best SVM classifier for this dataset         #
# Design the training parameters for the SVM.                               #
# What should the learning_rate be? What should C be?                       #
# What should num_iters be? Should X be scaled? Should X be kernelized?     #
#############################################################################
# your experiments below

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))


C = 1

svm.train(X,yy,learning_rate=1e-1,C=C,num_iters=8000,verbose=True)

#############################################################################
#  end of your code                                                         #
#############################################################################

#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(X)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected


yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1
test_pred = svm.predict(X_test)
print "Accuracy of model on test data is: ", metrics.accuracy_score(yy_test,test_pred)


#############################################################################
# Interpreting the coefficients of an SVM                                   #
# which words are the top predictors of spam?                               #
#############################################################################
# 4 lines of code expected

words, inv_words = utils.get_vocab_dict()

index = np.argsort(svm.theta)[-15:]
print "Top 15 predictors of spam are: "
for i in range(-1,-16,-1):
    print words[index[i]+1]


