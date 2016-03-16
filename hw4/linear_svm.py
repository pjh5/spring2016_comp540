import numpy as np

def svm_loss_twoclass(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ######################################################################
  # TODO                                                               #
  # Compute loss J and gradient of J with respect to theta             #
  # 2-3 lines of code expected                                         #
  ######################################################################
  yhx = y * X.dot(theta)
  J = np.mean(theta**2)/d + C*np.mean(np.maximum(1 - yhx, np.zeros([m,1])))
  grad = np.sum(theta / m - C*(yhx < 1)*np.mean(y*X, axis=0).T)

  ######################################################################
  # end of your code                                                   #
  ######################################################################
  return J, grad

