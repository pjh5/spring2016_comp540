import numpy as np
import fast_layers

def affine_forward(x, theta, theta_0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta_0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta_0)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected
  
  x1 = x.reshape(x.shape[0], -1)
  out = np.dot(x1, theta) + theta_0
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta_0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d, h)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta_0: Gradient with respect to b, of shape (h,)
  """
  x, theta, theta_0 = cache
  dx, dtheta, dtheta_0 = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected
  
  x1 = x.reshape(x.shape[0],-1)
  dx1 = np.dot(dout, theta.T)
  dx = dx1.reshape(x.shape)
  dtheta = np.dot(x1.T, dout)
  dtheta_0 = np.sum(dout, axis = 0)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta_0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1-2 lines of code expected.
  
  out = np.maximum(0, x)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1-2 lines of code expected. Hint: use np.where
  
  x1 = x
  x1[x1>0] = 1
  x1[x1<=0] = 0
  dx = np.multiply(dout, x1)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx




def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (K, C, HH, WW)
  - theta0: Biases, of shape (K,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, K, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # 14 lines of code expected
  
  m, C, H, W = x.shape
  K, _, HH, WW = theta.shape
  s = conv_param['stride']
  p = conv_param['pad']
  hi = 1 + (H + 2 * p - HH) / s
  wi = 1 + (W + 2 * p - WW) / s
  xp = np.lib.pad(x,((0,0), (0,0), (p,p), (p,p)),'constant',constant_values=0)
  out = np.zeros((m, K, hi, wi))
  
  for i in range(m):
    for j in range(K):
      for k in range(hi):
        for l in range(wi):
          out[i, j, k ,l] = np.sum(xp[i, :, k*s:k*s+HH, l*s:l*s+WW] * theta[j]) + theta0[j]
  """
  for i in range(hi):
    for j in range(wi):
      #print "xp", xp.shape
      #print "theta", theta.shape
      #print "theta0", theta0.shape
      #print "dot:", np.dot(theta[:, :,i, j].reshape(3,3,1,1).transpose(3,2,0,1), xp[:, :, i*s:i*s+HH-1, j*s:j*s+WW-1]).shape
      print (xp[:, :, i*s:i*s+HH, j*s:j*s+WW]).shape
      print theta.shape
      print out[:, :, i, j].shape
      print (np.einsum('abcd,xycd->abxy', xp[:, :, i*s:i*s+HH, j*s:j*s+WW], theta)).shape
      out[:, :, i, j] = np.einsum('abcd,xycd->ab', xp[:, :, i*s:i*s+HH, j*s:j*s+WW], theta) + theta0
        
        
  #print out
  #print out.shape
  """
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dtheta: Gradient with respect to theta
  - dtheta0: Gradient with respect to theta0
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # 20-22 lines of code expected
  
  x, theta, theta0, conv_param = cache
  m, C, H, W = x.shape
  K, _, HH, WW = theta.shape
  s = conv_param['stride']
  p = conv_param['pad']
  hi = 1 + (H + 2 * p - HH) / s
  wi = 1 + (W + 2 * p - WW) / s
  xp = np.lib.pad(x,((0,0), (0,0), (p,p), (p,p)), 'constant', constant_values=0)
  dxp = np.zeros_like(xp)
  dtheta = np.zeros_like(theta)
  dtheta0 = np.zeros_like(theta0)
  for i in range(m):
    for j in range(K):
      for k in range(hi):
        for l in range(wi):
          dxp[i, :, k*s:k*s+HH, l*s:l*s+WW] += theta[j] * dout[i, j, k, l]
          dtheta[j] += np.dot(xp[i, :, k*s:k*s+HH, l*s:l*s+WW], dout[i, j, k, l])
    dtheta0 += np.sum(dout[i], axis=(1,2))
  dx = dxp[:, :, p:-p, p:-p]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # 12-13 lines of code expected
  
  m, C, H, W = x.shape
  s = pool_param['stride']
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  hi = 1 + (H - ph) / s
  wi = 1 + (W - pw) / s
  out = np.zeros((m,C,hi,wi))
  for i in range(m):
    for j in range(C):
      for k in range(hi):
        for l in range(wi):
          out[i, j, k ,l] = np.max(x[i,j,k*s:k*s+ph,l*s:l*s+pw])
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # 15 lines of code expected
  
  x, pool_param = cache
  m, C, H, W = x.shape
  s = pool_param['stride']
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  hi = 1 + (H - ph) / s
  wi = 1 + (W - pw) / s
  dx = np.zeros_like(x)
  from numpy import unravel_index
  for i in range(m):
    for j in range(C):
      for k in range(hi):
        for l in range(wi):
          mh, mw = unravel_index(np.argmax(x[i,j,k*s:k*s+ph,l*s:l*s+pw]), x[i,j,k*s:k*s+ph,l*s:l*s+pw].shape)
          dx[i, j, k*s + mh , l*s + mw] += dout[i, j, k, l]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the output for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_output = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_output[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx

def norm_forward(x, gamma, beta, bn_param):
  
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  
  batch_size = 200
  
  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0, keepdims = True)
    sample_var = np.var(x, axis = 0, keepdims = True)
    norm = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * norm + beta
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    norm = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * norm + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  cache = x, norm, gamma, running_mean, running_var, eps
  return out, cache

def norm_backward(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x, norm, gamma, mean, var, eps = cache
  
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * norm, axis=0)
  
  dn = dout * gamma
  invstd = (var + eps)**-.5
  x -= mean
  dx = invstd * (dn - np.mean(dn, 0)) - invstd**3 * x * np.mean(dn * x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    dx = dout * mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx
