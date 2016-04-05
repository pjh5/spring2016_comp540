###################################################################################
#             Convolutional Neural Nets: a modular approach                       #
###################################################################################

###################################################################################
# So far we have worked with deep fully-connected networks, using them            #
# to explore different optimization strategies and network                        #
# architectures. Fully-connected networks are a good testbed for                  #
# experimentation because they are very computationally efficient, but            #
# in practice all state-of-the-art results use convolutional networks             #
#instead.  First you will implement several layer types that are used             #
# in convolutional networks. You will then use these layers to train a            #
# convolutional network on the CIFAR-10 dataset.                                  #
###################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
import fc_net 
from utils import get_CIFAR10_data
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
import solver
import cnn
from layer_utils import conv_relu_pool_forward, conv_relu_pool_backward
from layer_utils import conv_relu_forward, conv_relu_backward
import layers

###################################################################################
#  rel_error function useful for gradient checks                                  #
###################################################################################

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

###################################################################################
#   Load the (preprocessed) CIFAR10 data.                                         #
###################################################################################

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

# Problem 3.2.1
###################################################################################
# Convolution: Naive forward pass                                                 #
###################################################################################
# The core of a convolutional network is the convolution operation. In            #
# the file layers.py, implement the forward pass for the                          #
# convolution layer in the function conv_forward_naive.  You don't have           #
# to worry too much about efficiency at this point; just write the code           #
# in whatever way you find most clear.  You can test your implementation          #
# by running the following:                                                       #
###################################################################################

x_shape = (2, 3, 4, 4)
theta_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
theta = np.linspace(-0.2, 0.3, num=np.prod(theta_shape)).reshape(theta_shape)
theta0 = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = layers.conv_forward_naive(x, theta, theta0, conv_param)
correct_out = np.array([[[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]]])

# Compare your output to ours; difference should be around 1e-8
if out is not None:
  print 'Testing conv_forward_naive'
  print 'difference: ', rel_error(out, correct_out)

# Problem 3.2.2
###################################################################################
# Convolution: Naive backward pass                                                #
###################################################################################
# Implement the backward pass for the convolution operation in the                #
# function conv_backward_naive in the file layers.py. Again, you                  #
# don't need to worry too much about computational efficiency.  When you          #
# are done, run the following to check your backward pass with a numeric          #
# gradient check.                                                                 #
###################################################################################

x = np.random.randn(4, 3, 5, 5)
theta = np.random.randn(2, 3, 3, 3)
theta0 = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

if layers.conv_forward_naive(x,theta,theta0, conv_param)[0] is not None:
  dx_num = eval_numerical_gradient_array(lambda x: layers.conv_forward_naive(x, theta, theta0, conv_param)[0], x, dout)
  dtheta_num = eval_numerical_gradient_array(lambda theta: layers.conv_forward_naive(x, theta, theta0, conv_param)[0], theta, dout)
  dtheta0_num = eval_numerical_gradient_array(lambda theta0: layers.conv_forward_naive(x, theta, theta0, conv_param)[0], theta0, dout)

  out, cache = layers.conv_forward_naive(x, theta, theta0, conv_param)
  dx, dtheta, dtheta0 = layers.conv_backward_naive(dout, cache)

# Your errors should be around 1e-9
  print 'Testing conv_backward_naive function'
  print 'dx error: ', rel_error(dx, dx_num)
  print 'dtheta error: ', rel_error(dtheta, dtheta_num)
  print 'dtheta0 error: ', rel_error(dtheta0, dtheta0_num)

# Problem 3.2.3
###################################################################################
# Max pooling: Naive forward                                                      #
###################################################################################
# Implement the forward pass for the max-pooling operation in the                 #
# function max_pool_forward_naive in the file layers.py. Again,                   #
# don't worry too much about computational efficiency.  Check your                #
# implementation by running the following:                                        #
###################################################################################

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = layers.max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# Compare your output with ours. Difference should be around 1e-8.
if out is not None:
  print 'Testing max_pool_forward_naive function:'
  print 'difference: ', rel_error(out, correct_out)

# Problem 3.2.4
###################################################################################
# Max pooling: Naive backward                                                     #
###################################################################################
# Implement the backward pass for the max-pooling operation in the                #
# function max_pool_backward_naive in the file layers.py. You                     #
# don't need to worry about computational efficiency.  Check your                 #
# implementation with numeric gradient checking by running the                    #
# following:                                                                      #
###################################################################################

x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

if layers.max_pool_forward_naive(x,pool_param)[0] is not None:
  dx_num = eval_numerical_gradient_array(lambda x: layers.max_pool_forward_naive(x, pool_param)[0], x, dout)

  out, cache = layers.max_pool_forward_naive(x, pool_param)
  dx = layers.max_pool_backward_naive(dout, cache)

# Your error should be around 1e-12
  print 'Testing max_pool_backward_naive function:'
  print 'dx error: ', rel_error(dx, dx_num)

###################################################################################
# Fast layers                                                                     #
###################################################################################
# Making convolution and pooling layers fast can be challenging. To               #
# spare you the pain, we've provided fast implementations of the forward          #
# and backward passes for convolution and pooling layers in the file              #
# fast_layers.py.  The fast convolution implementation depends on                 #
# a Cython extension; to compile it you need to run the following from            #
# python setup.py build_ext --inplace The API for                                 #
# the fast versions of the convolution and pooling layers is exactly the          #
# same as the naive versions that you implemented above: the forward              #
# pass receives data, weights, and parameters and produces outputs and a          #
# cache object; the backward pass receives upstream derivatives and the           #
# cache object and produces gradients with respect to the data and                #
# weights.  NOTE: The fast implementation for pooling will only perform           #
# optimally if the pooling regions are non-overlapping and tile the               #
# input. If these conditions are not met then the fast pooling                    #
# implementation will not be much faster than the naive implementation.           #
# You can compare the performance of the naive and fast versions of               #
# these layers by running the following:                                          #
###################################################################################

from fast_layers import conv_forward_fast, conv_backward_fast
from time import time

x = np.random.randn(100, 3, 31, 31)
theta = np.random.randn(25, 3, 3, 3)
theta0 = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_naive, cache_naive = layers.conv_forward_naive(x, theta, theta0, conv_param)
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, theta, theta0, conv_param)
t2 = time()

if out_naive is not None: 
  print 'Testing conv_forward_fast:'
  print 'Naive: %fs' % (t1 - t0)
  print 'Fast: %fs' % (t2 - t1)
  print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
  print 'Difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive, dtheta_naive, dtheta0_naive = layers.conv_backward_naive(dout, cache_naive)
t1 = time()
dx_fast, dtheta_fast, dtheta0_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

if dx_naive is not None:
  print '\nTesting conv_backward_fast:'
  print 'Naive: %fs' % (t1 - t0)
  print 'Fast: %fs' % (t2 - t1)
  print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
  print 'dx difference: ', rel_error(dx_naive, dx_fast)
  print 'dtheta difference: ', rel_error(dtheta_naive, dtheta_fast)
  print 'dtheta0 difference: ', rel_error(dtheta0_naive, dtheta0_fast)

from fast_layers import max_pool_forward_fast, max_pool_backward_fast
from time import time

x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = layers.max_pool_forward_naive(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
t2 = time()

if out_naive is not None:
  print 'Testing pool_forward_fast:'
  print 'Naive: %fs' % (t1 - t0)
  print 'fast: %fs' % (t2 - t1)
  print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
  print 'difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive = layers.max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

if dx_naive is not None:
  print '\nTesting pool_backward_fast:'
  print 'Naive: %fs' % (t1 - t0)
  print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
  print 'dx difference: ', rel_error(dx_naive, dx_fast)

###################################################################################
# Convolutional "sandwich" layers                                                 #
###################################################################################
# Previously we introduced the concept of "sandwich" layers that combine          #
# multiple operations into commonly used patterns. In the file                    #
# layer_utils.py you will find sandwich layers that implement a                   #
# few commonly used patterns for convolutional networks.                          #
###################################################################################

x = np.random.randn(2, 3, 16, 16)
theta = np.random.randn(3, 3, 3, 3)
theta0 = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

if layers.conv_forward_naive(x,theta,theta0,conv_param)[0] is not None:
  out, cache = conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param)

  dx, dtheta, dtheta0 = conv_relu_pool_backward(dout, cache)
  dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param)[0], x, dout)
  dtheta_num = eval_numerical_gradient_array(lambda theta: conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param)[0], theta, dout)
  dtheta0_num = eval_numerical_gradient_array(lambda theta0: conv_relu_pool_forward(x, theta, theta0, conv_param, pool_param)[0], theta0, dout)

  print 'Testing conv_relu_pool'
  print 'dx error: ', rel_error(dx_num, dx)
  print 'dtheta error: ', rel_error(dtheta_num, dtheta)
  print 'dtheta0 error: ', rel_error(dtheta0_num, dtheta0)

x = np.random.randn(2, 3, 8, 8)
theta = np.random.randn(3, 3, 3, 3)
theta0 = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}

if layers.conv_forward_naive(x,theta,theta0,conv_param)[0] is not None:
  out, cache = conv_relu_forward(x, theta, theta0, conv_param)
  dx, dtheta, dtheta0 = conv_relu_backward(dout, cache)


  dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, theta, theta0, conv_param)[0], x, dout)
  dtheta_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, theta, theta0, conv_param)[0], theta, dout)
  dtheta0_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, theta, theta0, conv_param)[0], theta0, dout)

  print 'Testing conv_relu:'
  print 'dx error: ', rel_error(dx_num, dx)
  print 'dtheta error: ', rel_error(dtheta_num, dtheta)
  print 'dtheta0 error: ', rel_error(dtheta0_num, dtheta0)

# Problem 3.2.5
###################################################################################
# Three-layer Convnet                                                             #
###################################################################################
# Now that you have implemented all the necessary layers, we can put              #
# them together into a simple convolutional network.  Open the file               #
# cnn.py and complete the implementation of the ThreeLayerConvNet                 #
# class. Run the following cells to help you debug your implementation.           #
###################################################################################
# Sanity check loss                                                               #
#                                                                                 #
# After you build a new network, one of the first things you should do            #
# is sanity check the loss. When we use the softmax loss, we expect the           #
# loss for random weights (and no regularization) to be about log(C) for          #
# C classes. When we add regularization this should go up.                        #
###################################################################################

model = cnn.ThreeLayerConvNet()

m = 50
X = np.random.randn(m, 3, 32, 32)
y = np.random.randint(10, size=m)

if model.params != {}:
  loss, grads = model.loss(X, y)
  print 'Initial loss (no regularization): ', loss

  model.reg = 0.5
  loss, grads = model.loss(X, y)
  print 'Initial loss (with regularization): ', loss

###################################################################################
# Gradient check                                                                  #
#                                                                                 #
# After the loss looks reasonable, use numeric gradient checking to make          #
# sure that your backward pass is correct. When you use numeric gradient          #
# checking you should use a small amount of artifical data and a small            #
# number of neurons at each layer.                                                #
###################################################################################

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = cnn.ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)

if model.params != {}:
  loss, grads = model.loss(X, y)
  for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

###################################################################################
# Overfit small data                                                              #
#                                                                                 #
# A nice trick is to train your model with just a few training                    #
# samples. You should be able to overfit small datasets, which will               #
# result in very high training accuracy and comparatively low validation          #
# accuracy.                                                                       #
###################################################################################

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = cnn.ThreeLayerConvNet(weight_scale=1e-2)

if model.params != {}:
  asolver = solver.Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
  asolver.train()

  plt.subplot(2, 1, 1)
  plt.plot(asolver.loss_history, '-o')
  plt.xlabel('iteration')
  plt.ylabel('loss')

  plt.subplot(2, 1, 2)
  plt.plot(asolver.train_acc_history, '-o')
  plt.plot(asolver.val_acc_history, '-o')
  plt.legend(['train', 'val'], loc='upper left')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.show()

###################################################################################
# Train the net                                                                   #
#                                                                                 #
# By training the three-layer convolutional network for one epoch, you            #
# should achieve greater than 40% accuracy on the training set:                   #
###################################################################################

model = cnn.ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

if model.params != {}:
  asolver = solver.Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=1)
  asolver.train()

###################################################################################
# Visualize Filters                                                               #
#                                                                                 #
# You can visualize the first-layer convolutional filters from the trained        #
# network by running the following:                                               #
###################################################################################

from vis_utils import visualize_grid

if model.params != {}:
  grid = visualize_grid(model.params['theta1'].transpose(0, 2, 3, 1))
  plt.imshow(grid.astype('uint8'))
  plt.axis('off')
  plt.gcf().set_size_inches(5, 5)
  plt.show()

# Problem 3.2.6
###################################################################################
# Decrease the size of the filter and increase number of filters                  #
# The aim is to achieve > 50% validation accuracy                                 #
###################################################################################
# TODO: build your model, set up a solver, train the model, visualize the weights
