import scipy.io
import utils
import numpy as np
from sklearn import linear_model

# No modifications in this script
# complete the functions in util.py; then run the script

# load the spam data in

Xtrain,Xtest,ytrain,ytest = utils.load_spam_data()

# Preprocess the data 

Xtrain_std,mu,sigma = utils.std_features(Xtrain)
Xtrain_logt = utils.log_features(Xtrain)
Xtrain_bin = utils.bin_features(Xtrain)

Xtest_std = (Xtest - mu)/sigma
Xtest_logt = utils.log_features(Xtest)
Xtest_bin = utils.bin_features(Xtest)

# find good lambda by cross validation for these three sets

def run_dataset(X,ytrain,Xt,ytest,type,penalty):

    best_lambda = utils.select_lambda_crossval(X,ytrain,0.1,5.1,0.5,penalty)
    print "best_lambda = ", best_lambda

    # train a classifier on best_lambda and run it
    if penalty == "l2":
        lreg = linear_model.LogisticRegression(penalty=penalty,C=1.0/best_lambda, solver='lbfgs',fit_intercept=True)
    else:
        lreg = linear_model.LogisticRegression(penalty=penalty,C=1.0/best_lambda, solver='liblinear',fit_intercept=True)
    lreg.fit(X,ytrain)
    print "Coefficients = ", lreg.intercept_,lreg.coef_
    predy = lreg.predict(Xt)
    print "Accuracy on set aside test set for ", type, " = ", np.mean(predy==ytest)

print "L2 Penalty experiments -----------"
run_dataset(Xtrain_std,ytrain,Xtest_std,ytest,"std","l2")
run_dataset(Xtrain_logt,ytrain,Xtest_logt,ytest,"logt","l2")
run_dataset(Xtrain_bin,ytrain,Xtest_bin,ytest,"bin","l2")

print "L1 Penalty experiments -----------"
run_dataset(Xtrain_std,ytrain,Xtest_std,ytest,"std","l1")
run_dataset(Xtrain_logt,ytrain,Xtest_logt,ytest,"logt","l1")
run_dataset(Xtrain_bin,ytrain,Xtest_bin,ytest,"bin","l1")
