from layers import *

def affine_relu_forward(x, theta, theta_0):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - theta, theta_0: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, theta, theta_0)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dtheta, dtheta_0 = affine_backward(da, fc_cache)
  return dx, dtheta, dtheta_0


from fast_layers import *

def conv_relu_forward(x, theta, theta_0, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - theta, theta_0, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, theta, theta_0, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dtheta, dtheta_0 = conv_backward_fast(da, conv_cache)
  return dx, dtheta, dtheta_0


def conv_relu_pool_forward(x, theta, theta_0, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - theta, theta_0, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, theta, theta_0, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dtheta, dtheta_0 = conv_backward_fast(da, conv_cache)
  return dx, dtheta, dtheta_0

def conv_relu_pool_drop_forward(x, theta, theta_0, conv_param, pool_param, drop_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - theta, theta_0, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, theta, theta_0, conv_param)
  s, relu_cache = relu_forward(a)
  p, pool_cache = max_pool_forward_fast(s, pool_param)
  out, drop_cache = dropout_forward(p, drop_param)
  cache = (conv_cache, relu_cache, pool_cache, drop_cache)
  return out, cache


def conv_relu_pool_drop_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache, drop_cache = cache
  dd = dropout_backward(dout, drop_cache)
  ds = max_pool_backward_fast(dd, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dtheta, dtheta_0 = conv_backward_fast(da, conv_cache)
  return dx, dtheta, dtheta_0


def conv_relu_pool_norm_forward(x, theta, theta_0, conv_param, pool_param, mode, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - theta, theta_0, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, theta, theta_0, conv_param)
  s, relu_cache = relu_forward(a)
  p, pool_cache = max_pool_forward_fast(s, pool_param)
  out, norm_cache = norm_forward(p, gamma, beta, bn_param)
  cache = (conv_cache, relu_cache, pool_cache, norm_cache)
  return out, cache


def conv_relu_pool_norm_backward(dout, cache, mode):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache, norm_cache = cache
  dn = norm_backward(dout, norm_cache)
  ds = max_pool_backward_fast(dn, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dtheta, dtheta_0 = conv_backward_fast(da, conv_cache)
  return dx, dtheta, dtheta_0

