import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC

##########################################################################
# Utilities for computing kernels                                        #
##########################################################################

def gaussian_kernel(x1,x2,sigma):
    k = 0

    #######################################################################
    # TODO                                                                #
    # Compute Gaussian kernel                                             #
    # 1 line of code expected                                             #
    #######################################################################
    k = np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))

    #######################################################################
    #  end of code                                                        #
    #######################################################################

    return k
    
def linear_kernel(x1,x2):
  return np.linalg.norm(x1-x2)

def dot_product_kernel(x1,x2):
  return x1.dot(x2)

def polynomial_kernel(x1,x2,sigma,d):
    k = 0

    #######################################################################
    # TODO                                                                #
    # Compute Gaussian kernel                                             #
    # 1 line of code expected                                             #
    #######################################################################
    k = (1 + x1.dot(x2)/sigma)**d

    #######################################################################
    #  end of code                                                        #
    #######################################################################

    return k
    
def chi_kernel(x1,x2,sigma=0.5):

    #######################################################################
    # TODO                                                                #
    # Compute Gaussian kernel                                             #
    # 1 line of code expected                                             #
    #######################################################################
    
    k = chi2_kernel(x1, x2, gamma=sigma)

    #######################################################################
    #  end of code                                                        #
    #######################################################################

    return k

def sigmoid_kernel(x1,x2,sigma,tt):
    k = 0

    #######################################################################
    # TODO                                                                #
    # Compute Gaussian kernel                                             #
    # 1 line of code expected                                             #
    #######################################################################
    k = np.tanh(x1.dot(x2)/sigma - tt)

    #######################################################################
    #  end of code                                                        #
    #######################################################################

    return k

##########################################################################
# Utilities for loading mat files                                        #
##########################################################################
    
def load_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    return X,y

def loadval_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    X_val = data['Xval']
    y_val = data['yval'].flatten()
    return X,y, X_val, y_val

##########################################################################
# Utilities for plotting data and decision boundaries                    #
##########################################################################

def plot_twoclass_data(X,y,xlabel,ylabel,legend):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_adjustable('box')
<<<<<<< HEAD
    X0 = X[np.where(y<=0)]
=======
    X0 = X[np.where(y==-1)]
>>>>>>> 9c1e7b13a835127477b69825a1e78e6a78e4fc1b
    X1 = X[np.where(y==1)]
    plt.scatter(X0[:,0],X0[:,1],c='red', s=80, label = legend[0])
    plt.scatter(X1[:,0],X1[:,1],c='green', s = 80, label=legend[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    return ax

def plot_decision_boundary_sklearn(X,y,clf,  xlabel, ylabel, legend):

    plot_twoclass_data(X,y,xlabel,ylabel,legend)

    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh
    Z = np.array(clf.predict(np.c_[xx1.ravel(), xx2.ravel()]))

    # Put the result into a contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def plot_decision_boundary(X,y,clf,xlabel, ylabel, legend):

    plot_twoclass_data(X,y,xlabel,ylabel,legend)
    
    # create a mesh to plot in
    h = 0.01
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # make predictions on this mesh (but add intercept term)
    Z = np.array(clf.predict(np.c_[np.ones((xx1.ravel().shape[0],)), xx1.ravel(), xx2.ravel()]))

    # Put the result into a contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


def plot_decision_kernel_boundary(X,y,scaler, sigma, clf,  xlabel, ylabel, legend):

    ax = plot_twoclass_data(X,y,xlabel,ylabel,legend)
    ax.autoscale(False)

    # create a mesh to plot in
    h = 0.05
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    ZZ = np.array(np.c_[xx1.ravel(), xx2.ravel()])
    K = np.array([gaussian_kernel(x1,x2,sigma) for x1 in ZZ for x2 in X]).reshape((ZZ.shape[0],X.shape[0]))

    # need to scale it
    scaleK = scaler.transform(K)

    # and add the intercept column of ones
    KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK.T]).T

    # make predictions on this mesh
    Z = clf.predict(KK)

    # Put the result into a contour plot
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])


##########################################################################
# Utilities for reading vocabulary list                                  #
##########################################################################

# words: maps index to actual word
# inv_words: maps word to index in vocab list

def get_vocab_dict():
    words = {}
    inv_words = {}
    f = open('data/vocab.txt','r')
    for line in f:
        if line != '':
            (ind,word) = line.split('\t')
            words[int(ind)] = word.rstrip('\n')
            inv_words[word.rstrip('\n')] = int(ind)
    return words, inv_words
