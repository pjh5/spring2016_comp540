import numpy as np

class LinearClassifier(object):

  def __init__(self):
    self.theta = None

  def train(self, X, y, learning_rate=1e-3, C=1e-5, num_iters=100,
             verbose=False):
    """
    Train this linear classifier using gradient descent.

    Inputs:
    - X: A numpy array of shape (m, d) containing training data; there are m
      training samples each of dimension d.
    - y: A numpy array of shape (m,) containing training labels; y[i] = -1,1
    - learning_rate: (float) learning rate for optimization.
    - C: (float) penalty tem.
    - num_iters: (integer) number of steps to take when optimizing
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    m,d = X.shape

    if self.theta is None:
      # lazily initialize theta
      self.theta = np.zeros((d,))

    # Run gradient descent to optimize theta
    loss_history = []
    for it in xrange(num_iters):

      # evaluate loss and gradient
      loss, grad = self.loss(X, y, C)
      loss_history.append(loss)

      # perform parameter update

      self.theta = self.theta - learning_rate * grad

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the coefficients of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: m x d array of training data. Each row is a d-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length m, and each element is an integer (+1,-1) giving the
      predicted class.
    """
    y_pred = np.dot(X,self.theta)
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    
    return y_pred
  
  def loss(self, X, y, C):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X: A numpy array of shape (m,d)
    - y: A numpy array of shape (m,) containing labels for X.

    - C: (float) penalty term.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.theta; an array of the same shape as theta
    """
    pass

from linear_svm import svm_loss_twoclass

class LinearSVM_twoclass(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X, y, C):
    return svm_loss_twoclass(self.theta, X, y, C)


