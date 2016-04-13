import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *
import pickle

class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  (conv - relu - pool) x 3 - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (m, C, H, W)
  consisting of m images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[96, 192, 384], filter_size=[3, 3, 3],
               hidden_dim=[1000, 1000], num_classes=10, weight_scale=1e-3, reg=0.0,load=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys      #
    # 'theta1' and 'theta1_0'; use keys 'theta2' and 'theta2_0' for the        #
    # weights and biases of the hidden affine layer, and keys 'theta3' and     #
    # 'theta3_0' for the weights and biases of the output affine layer.        #
    ############################################################################
    # about 15 lines of code
    
    
    if load:
      self.params = pickle.load(open( "cnn3.params", "rb" ))
    else:
      conv_param = {'stride': 1, 'pad': 1}
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      
      self.params['theta1'] = np.random.randn(num_filters[0], input_dim[0], filter_size[0], filter_size[0]) * weight_scale
      self.params['theta1_0'] = np.zeros(num_filters[0])
      t2 = 1 + ((1 + (input_dim[1] + 2 * conv_param['pad'] - filter_size[0]) / conv_param['stride']) - pool_param['pool_height']) / pool_param['stride']
      
      self.params['theta2'] = np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1]) * weight_scale
      self.params['theta2_0'] = np.zeros(num_filters[1])
      
      t3 = 1 + ((1 + (t2 + 2 * conv_param['pad'] - filter_size[1]) / conv_param['stride']) - pool_param['pool_height']) / pool_param['stride']
      
      self.params['theta3'] = np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2]) * weight_scale
      self.params['theta3_0'] = np.zeros(num_filters[2])
      
      t4 = 1 + ((1 + (t3 + 2 * conv_param['pad'] - filter_size[2]) / conv_param['stride']) - pool_param['pool_height']) / pool_param['stride']
      
      self.params['theta4'] = np.random.randn(num_filters[2] * t4 * t4, hidden_dim[0]) * weight_scale
      self.params['theta4_0'] = np.zeros(hidden_dim[0])
      
      self.params['theta5'] = np.random.randn(hidden_dim[0], hidden_dim[1]) * weight_scale
      self.params['theta5_0'] = np.zeros(hidden_dim[1])
      
      self.params['theta6'] = np.random.randn(hidden_dim[1], num_classes) * weight_scale
      self.params['theta6_0'] = np.zeros(num_classes)
      pickle.dump( self.params, open( "cnn3.params", "wb" ))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    #bn_param[mode] = mode
    
    drop_param={'mode': mode, 'p': 0.5}
    
    #X = augment(X)
    theta1, theta1_0 = self.params['theta1'], self.params['theta1_0']
    theta2, theta2_0 = self.params['theta2'], self.params['theta2_0']
    theta3, theta3_0 = self.params['theta3'], self.params['theta3_0']
    theta4, theta4_0 = self.params['theta4'], self.params['theta4_0']
    theta5, theta5_0 = self.params['theta5'], self.params['theta5_0']
    theta6, theta6_0 = self.params['theta6'], self.params['theta6_0']
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': 1, 'pad': 1}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    
    a1, c1 = conv_relu_pool_drop_forward(X, theta1, theta1_0, conv_param, pool_param, drop_param)
    a2, c2 = conv_relu_pool_drop_forward(a1, theta2, theta2_0, conv_param, pool_param, drop_param)
    a3, c3 = conv_relu_pool_drop_forward(a2, theta3, theta3_0, conv_param, pool_param, drop_param)
    a4, c4 = affine_relu_forward(a3, theta4, theta4_0)
    a5, c5 = affine_forward(a4, theta5, theta5_0)
    a6, c6 = affine_forward(a5, theta6, theta6_0)
    scores = a6
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code.
    
    loss, d6 = softmax_loss(scores, y)
    loss += self.reg * np.sum(self.params['theta1']**2) / 2
    loss += self.reg * np.sum(self.params['theta2']**2) / 2
    loss += self.reg * np.sum(self.params['theta3']**2) / 2
    loss += self.reg * np.sum(self.params['theta4']**2) / 2
    loss += self.reg * np.sum(self.params['theta5']**2) / 2
    loss += self.reg * np.sum(self.params['theta6']**2) / 2
    
    d5, grads['theta6'], grads['theta6_0'] = affine_backward(d6, c6)
    d4, grads['theta5'], grads['theta5_0'] = affine_backward(d5, c5)
    d3, grads['theta4'], grads['theta4_0'] = affine_relu_backward(d4, c4)
    d2, grads['theta3'], grads['theta3_0'] = conv_relu_pool_drop_backward(d3, c3)
    d1, grads['theta2'], grads['theta2_0'] = conv_relu_pool_drop_backward(d2, c2)
    d0, grads['theta1'], grads['theta1_0'] = conv_relu_pool_drop_backward(d1, c1)
    
    grads['theta6'] += self.reg * self.params['theta6']
    grads['theta5'] += self.reg * self.params['theta5']
    grads['theta4'] += self.reg * self.params['theta4']
    grads['theta3'] += self.reg * self.params['theta3']
    grads['theta2'] += self.reg * self.params['theta2']
    grads['theta1'] += self.reg * self.params['theta1']
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
  def predict(self, X):
    
    filter_size = theta1.shape[2]
    conv_param = {'stride': 1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    drop_param={'mode': 'test', 'p': 0.5}
    scores = None
    
    a1, c1 = conv_relu_pool_drop_forward(X, theta1, theta1_0, conv_param, pool_param, drop_param)
    a2, c2 = conv_relu_pool_drop_forward(a1, theta2, theta2_0, conv_param, pool_param, drop_param)
    a3, c3 = conv_relu_pool_drop_forward(a2, theta3, theta3_0, conv_param, pool_param, drop_param)
    a4, c4 = affine_relu_forward(a3, theta4, theta4_0)
    a5, c5 = affine_forward(a4, theta5, theta5_0)
    a6, c6 = affine_forward(a5, theta6, theta6_0)
    scores = a6
    
    pred = np.argmax(scores, axis=0)
    
    return pred
  
  def save(self):
    pickle.dump( self.params, open( "cnn3.params", "wb" ))

  def augment(x):
    out = x
    return out
"""
    for i in range(x.shape[0]):
      angle = np.random.randint(0, 40, 1)[0] - 20
      c = np.float(np.random.randint(0, 255, 1)[0])
      resize = 1 + np.float(np.random.randint(0, 40, 1)[0]) / 100
      out[i] = rescale(x[i], resize)
      if out[i].shape[1] < 32:
        d = 32 - au.shape[1]
        au = np.lib.pad(au,((d,d),(d,d),(0,0)),'constant',constant_values=c)
      c = np.float(np.random.randint(0, 255, 1)[0])
      au = rotate(au, angle, resize=False, mode='constant', cval=c) / 255
      b = au.shape[1] - 32
      if b > 0:
        h = np.random.randint(0, b, 1)[0]
        w = np.random.randint(0, b, 1)[0]
      else:
        h = 0 
        w = 0 
      out[i] = au[h:32+h, w:32+w]
      #print au.shape
"""