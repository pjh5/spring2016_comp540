from sklearn import preprocessing, metrics, cross_validation
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass
import sys


#############################################################################
# load the SPAM email training and test dataset                             #
#############################################################################

print "Reading the data..."
sys.stdout.flush()
X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1

X, Xval, yy, yyval = cross_validation.train_test_split(X, yy, test_size=0.1)

test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()
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


Cvals = [0.01,0.03,0.1,0.3,1,3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,1,3,10,30]

for sigma in sigma_vals:
  print "Calculating K"
  sys.stdout.flush()
  K = np.array([utils.gaussian_kernel(x1,x2,sigma) for x1 in X for x2 in X]).reshape(X.shape[0],X.shape[0])
  scaler = preprocessing.StandardScaler().fit(K)
  scaleK = scaler.transform(K)
  KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK]).T
  
  Kval = np.array([utils.gaussian_kernel(x1,x2,sigma) for x1 in Xval for x2 in X]).reshape(Xval.shape[0], X.shape[0])
  scaleKval = scaler.transform(Kval)
  KKval = np.vstack([np.ones((scaleKval.shape[0],)),scaleKval.T]).T
  print "Done!"
  for C in Cvals:
    print "sigma=", sigma, ", C=", C
    sys.stdout.flush()
    svm.theta = np.zeros((KK.shape[1],))
    svm.train(KK,yy,learning_rate=1e-4,C=C,num_iters=20000)
    
    y_pred = svm.predict(KKval)
    acc = metrics.accuracy_score(yyval,y_pred)
    
    if (acc > best_acc):
      best_acc = acc
      best_C = C
      best_sigma = sigma
print "Best C is ", best_C, ", best sigma is ", best_sigma
sys.stdout.flush()

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


