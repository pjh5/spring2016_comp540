import numpy as np
from random import shuffle
import scipy.sparse


class SoftmaxClassifier:

  def __init__(self):
    self.theta = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=4000,
            batch_size=400, verbose=False):
    """
    Train the classifier using mini-batch stochastic gradient descent.

    Inputs:
    - X: m x d array of training data. Each training point is a d-dimensional
         row.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train,dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    self.theta = np.random.randn(dim,num_classes) * 0.001

    loss_history = []
    for it in range(num_iters):
      X_batch = np.zeros(batch_size)
      y_batch = np.zeros(batch_size)

      ind = np.random.randint(0, num_train, batch_size)
      X_batch = X[ind,:]
      y_batch = y[ind]
	  
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
	  
      self.theta -= grad * learning_rate

      if verbose and it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: m x d array of training data. Each row is a d-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length m, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])

    y_pred = np.argmax(X.dot(self.theta),axis=1)
	
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: m x d array of data; each row is a data point.
    - y_batch: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.theta; an array of the same shape as theta
    """

    return softmax_loss_vectorized(self.theta, X_batch, y_batch, reg)

  
def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  for i in np.arange(m):
    J -= np.log(np.exp(X[i,:].dot(theta[:,y[i]])) / np.sum(np.exp(X[i,:].dot(theta))))
  J = (J  + reg * np.sum(theta**2)) / float(m)
  for i in np.arange(m):
    for k in np.arange(theta.shape[1]):
      grad[:,k] -= X[i,:] * ((y[i] == k) - np.exp(X[i,:].dot(theta[:,k])) / np.sum(np.exp(X[i,:].dot(theta)))) / float(m)
	  
  return J, grad

def convert_y_to_matrix(y):
  """
  convert an array of m elements with values in {0,...,K-1} to a boolean matrix
  of size m x K where there is a 1 for the value of y in that row.

  """
  y = np.array(y)
  data = np.ones(len(y))
  indptr = np.arange(len(y)+1)
  mat = scipy.sparse.csr_matrix((data,y,indptr))
  return mat.todense()

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape
  
  Xtheta = X.dot(theta)
  XthetaRed = (Xtheta.T - np.max(Xtheta, axis=1)).T
  prob = np.exp(XthetaRed.T) / np.sum(np.exp(XthetaRed), axis=1)
  yisk = (y.reshape([m,1]) == np.arange(theta.shape[1]))
  J = (-np.sum(np.multiply(yisk, np.log(prob).T)) + reg * np.sum(theta**2) / 2 ) / m
  grad = (-(yisk.T - prob).dot(X).T + reg * theta) / float(m)

  return J, grad
