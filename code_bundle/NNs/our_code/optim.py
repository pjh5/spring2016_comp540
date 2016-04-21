import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(theta, dtheta, config=None):

Inputs:
  - theta: A numpy array giving the current weights.
  - dtheta: A numpy array of the same shape as theta giving the gradient of the
    loss with respect to theta.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_theta: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating theta and
setting next_theta equal to theta.
"""


def sgd(theta, dtheta, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  theta -= config['learning_rate'] * dtheta
  return theta, config


def sgd_momentum(theta, dtheta, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(theta))
  
  next_theta = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_theta variable. You should also use and update the velocity v.   #
  #############################################################################
  # 2 lines of code expected
  
  v = config['momentum'] * v - config['learning_rate'] * dtheta
  next_theta = theta + v
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  config['velocity'] = v

  return next_theta, config



def rmsprop(theta, dtheta, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(theta))

  next_theta = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of     #
  # theta in the next_theta variable. Don't forget to update cache value      #  
  # stored in config['cache'].                                                #
  #############################################################################
  # 2 lines of code expected
  
  config['cache'] = config['cache'] * config['decay_rate'] + (1 - config['decay_rate']) * dtheta * dtheta
  next_theta = theta - config['learning_rate'] * dtheta / (np.sqrt(config['cache']) + config['epsilon'])
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return next_theta, config


def adam(theta, dtheta, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(theta))
  config.setdefault('v', np.zeros_like(theta))
  config.setdefault('t', 0)
  
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dtheta
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dtheta**2)
  config['t'] += 1

  mt_hat = config['m'] / (1 - (config['beta1'])**config['t'])
  vt_hat = config['v'] / (1 - (config['beta2'])**config['t'])
  next_theta = theta - config['learning_rate'] * mt_hat / (np.sqrt(vt_hat + config['epsilon']))
  
  return next_theta, config
