from sklearn import preprocessing, metrics, cross_validation
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass
import sys


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

print "-------3RD DEGREE POLYNOMIAL KERNEL-------"
print "Reading the data..."
sys.stdout.flush()
X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1

X, Xval, yy, yyval = cross_validation.train_test_split(X, yy, test_size=0.1)
test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()
yy_test = np.ones(y_test.shape)
yy_test[y_test==0] = -1
print "Done!"
sys.stdout.flush()

#############################################################################
# your code for setting up the best SVM classifier for this dataset         #
# Design the training parameters for the SVM.                               #
# What should the learning_rate be? What should C be?                       #
# What should num_iters be? Should X be scaled? Should X be kernelized?     #
#############################################################################
# your experiments below

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))

it = 500
lr = 1e-4
print it," iters"
print "learning rate ",lr

Cvals = [0.15] #[0.15, 0.175, 0.2, 0.25, 0.3]
sigma_vals = [0.1] #[0.1, 0.5, 1, 5 , 10]

#best_C = .15
best_C = 0.15
#best_sigma = .1
best_sigma = 0.1

best_acc = 0;
acc_mat_val=np.zeros((len(Cvals), len(sigma_vals)))
acc_mat_train=np.zeros((len(Cvals), len(sigma_vals)))
for sigma in sigma_vals:
  print "Calculating K"
  sys.stdout.flush()
  K = np.array([utils.polynomial_kernel(x1,x2,sigma,3) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])
  scaler = preprocessing.StandardScaler().fit(K)
  scaleK = scaler.transform(K)
  KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T
  
  Kval = np.array([utils.polynomial_kernel(x1,x2,sigma,3) for x1 in Xval for x2 in X]).reshape(Xval.shape[0], X.shape[0])
  scaleKval = scaler.transform(Kval)
  KKval = np.vstack([np.ones((scaleKval.shape[0],)),scaleKval.T]).T
  print "Done!"
  for C in Cvals:
    print "sigma=", sigma, ", C=", C
    sys.stdout.flush()
    svm.theta = np.zeros((KK.shape[1],))
    svm.train(KK,yy,learning_rate=lr,C=C,num_iters=it)
    
    y_train_pred = svm.predict(KK)
    acc = metrics.accuracy_score(yy,y_train_pred)
    print "C=", C, "sigma=", sigma, "TrainAccuracy:", acc
    acc_mat_train[np.where(np.array(Cvals)==C)[0][0], np.where(np.array(sigma_vals)==sigma)[0][0]] = acc
    
    y_pred = svm.predict(KKval)
    acc = metrics.accuracy_score(yyval,y_pred)
    print "C=", C, "sigma=", sigma, "ValAccuracy:", acc
    
    acc_mat_val[np.where(np.array(Cvals)==C)[0][0], np.where(np.array(sigma_vals)==sigma)[0][0]] = acc
    if (acc > best_acc):
      best_acc = acc
      best_C = C
      best_sigma = sigma

print "Best C is ", best_C, ", best sigma is ", best_sigma
sys.stdout.flush()

print "C range:"
print Cvals

print "Sigma range:"
print sigma_vals

print "Training acc:"
print acc_mat_train

print "Validation acc:"
print acc_mat_val

sys.stdout.flush()

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))
K = np.array([utils.polynomial_kernel(x1,x2,best_sigma,3) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])
scaler = preprocessing.StandardScaler().fit(K)
scaleK = scaler.transform(K)
KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T
svm.theta = np.zeros((KK.shape[1],))
K_test = np.array([utils.polynomial_kernel(x1,x2,best_sigma,3) for x1 in X_test for x2 in X]).reshape(X_test.shape[0],X.shape[0])
scaler_test = preprocessing.StandardScaler().fit(K_test)
scaleK_test = scaler_test.transform(K_test)
KK_test = np.vstack([np.ones((scaleK_test.shape[0],)),scaleK_test.T]).T
svm.train(KK,yy,learning_rate=lr,C=best_C,num_iters=it)

#############################################################################
#  end of your code                                                         #
#############################################################################

#############################################################################
# what is the accuracy of the best model on the training data itself?       #
#############################################################################
# 2 lines of code expected

y_pred = svm.predict(KK)
print "Accuracy of model on training data is: ", metrics.accuracy_score(yy,y_pred)


#############################################################################
# what is the accuracy of the best model on the test data?                  #
#############################################################################
# 2 lines of code expected

test_pred = svm.predict(KK_test)
print "Accuracy of model on test data is: ", metrics.accuracy_score(yy_test, test_pred)


#############################################################################
# Interpreting the coefficients of an SVM                                   #
# which words are the top predictors of spam?                               #
#############################################################################
# 4 lines of code expected

words, inv_words = utils.get_vocab_dict()
index = np.dot(svm.theta[1:],X).argsort()[-15:]
print index
print "Top 15 predictors of spam are: "
for i in range(len(index)):
    print words[index[i]]


