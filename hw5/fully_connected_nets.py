###################################################################################
#             Fully-Connected Neural Nets: a modular approach                     #
###################################################################################

###################################################################################
# Ideally we want to build networks using a modular design so that we             #
# can implement different layer types in isolation and then snap them             #
# together into models with different architectures.  In this exercise            #
# we will implement fully-connected networks using a  modular                     #
# approach. For each layer we will implement a forward and a backward             #
# function. The forward function will receive inputs, weights, and other          #
# parameters and will return both an output and a cache object storing            #
# data needed for the backward pass.                                              #
###################################################################################

import time
import numpy as np
import  utils
import matplotlib.pyplot as plt
import layers
import fc_net 
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
import solver
import layer_utils

###################################################################################
#  rel_error function useful for gradient checks                                  #
###################################################################################

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

###################################################################################
#   Load the (preprocessed) CIFAR10 data.                                         #
###################################################################################

data = utils.get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape


# Problem 3.1.1
###################################################################################
#   Affine layer: forward.                                                        #
###################################################################################
#   In the file layers.py implement the affine_forward function.                  #
#   Once you are done you can test your implementation by running the following:  #
###################################################################################

# Test the affine_forward function

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
theta_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
theta = np.linspace(-0.2, 0.3, num=theta_size).reshape(np.prod(input_shape), output_dim)
theta_0 = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = layers.affine_forward(x, theta, theta_0)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# Compare your output with ours. The error should be around 1e-9.
if out:
  print 'Testing affine_forward function:'
  print 'difference (should be around 1e-9): ', rel_error(out, correct_out)

# Problem 3.1.2
###################################################################################
#   Affine layer: backward.                                                       #
###################################################################################
#   In the file layers.py implement the affine_backward function.                 #
#   Once you are done you can test your implementation using numeric gradient.    #
###################################################################################

# Test the affine_backward function

x = np.random.randn(10, 2, 3)
theta = np.random.randn(6, 5)
theta_0 = np.random.randn(5)
dout = np.random.randn(10, 5)

if layers.affine_forward(x,theta,theta_0)[0] is not None:
  dx_num = eval_numerical_gradient_array(lambda x: layers.affine_forward(x, theta, theta_0)[0], x, dout)
  dtheta_num = eval_numerical_gradient_array(lambda theta: layers.affine_forward(x, theta, theta_0)[0], theta, dout)
  dtheta_0_num = eval_numerical_gradient_array(lambda b: layers.affine_forward(x, theta, theta_0)[0], theta_0, dout)

  _, cache = layers.affine_forward(x, theta, theta_0)
  dx, dtheta, dtheta_0 = layers.affine_backward(dout, cache)

# The error should be around 1e-10
  print 'Testing affine_backward function:'
  print 'dx error (should be around 1e-10): ', rel_error(dx_num, dx)
  print 'dtheta error (should be around 1e-10): ', rel_error(dtheta_num, dtheta)
  print 'dtheta_0 error (should be around 1e-10): ', rel_error(dtheta_0_num, dtheta_0)


# Problem 3.1.3
###################################################################################
#   ReLU layer: forward                                                           #
###################################################################################
#   In the file layers.py implement the forward pass for the ReLU activation in   #
#   the relu_forward function.                                                    #
#   Once you are done you can test your implementation using the following.       #
###################################################################################

# Test the relu_forward function

x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out, _ = layers.relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# Compare your output with ours. The error should be around 1e-8

if out is not None:
  print 'Testing relu_forward function:'
  print 'difference (should be around 1e-8): ', rel_error(out, correct_out)

# Problem 3.1.4
###################################################################################
#   ReLU layer: backward                                                          #
###################################################################################
#   In the file layers.py implement the backward pass for the ReLU activation in  #
#   the relu_backward function.                                                   #
#   Once you are done you can test your implementation using the numeric gradient #
#   checking.                                                                     #
###################################################################################

# Test the relu_backward function

x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

if layers.relu_forward(x)[0] is not None:
  dx_num = eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)

  _, cache = layers.relu_forward(x)
  dx = layers.relu_backward(dout, cache)

# The error should be around 1e-12
  print 'Testing relu_backward function:'
  print 'dx error: (should be around 1e-12): ', rel_error(dx_num, dx)


###################################################################################
#   Sandwich layers                                                               #
###################################################################################
#   There are some common patterns of layers that are frequently used in          #
#   neural nets. For example, affine layers are frequently followed by a          #
#   ReLU nonlinearity. To make these common patterns easy, we define              #
#   several convenience layers in the file layer_utils.py.  For now               #
#   take a look at the affine_relu_forward and affine_relu_backward               #
#   functions, and run the following to numerically gradient check the            #
#   backward pass.                                                                #
###################################################################################

x = np.random.randn(2, 3, 4)
theta = np.random.randn(12, 10)
theta_0 = np.random.randn(10)
dout = np.random.randn(2, 10)

if layers.affine_forward(x,theta,theta_0)[0] is not None:
  out, cache = layer_utils.affine_relu_forward(x, theta, theta_0)
  dx, dtheta, dtheta_0 = layer_utils.affine_relu_backward(dout, cache)

  dx_num = eval_numerical_gradient_array(lambda x: layer_utils.affine_relu_forward(x, theta, theta_0)[0], x, dout)
  dtheta_num = eval_numerical_gradient_array(lambda w: layer_utils.affine_relu_forward(x, theta, theta_0)[0], theta, dout)
  dtheta_0_num = eval_numerical_gradient_array(lambda b: layer_utils.affine_relu_forward(x, theta, theta_0)[0], theta_0, dout)

  print 'Testing affine_relu_forward:'
  print 'dx error: ', rel_error(dx_num, dx)
  print 'dtheta error: ', rel_error(dtheta_num, dtheta)
  print 'dtheta_0 error: ', rel_error(dtheta_0_num, dtheta_0)


###################################################################################
#   Loss layers: Softmax and SVM                                                  #
###################################################################################
#   You implemented these loss functions in the last assignment, so we'll         #
#   give them to you for free here. You should still make sure you                #
#   understand how they work by looking at the implementations in                 #
#   layers.py.  You can make sure that the implementations are                    #
#   correct by running the following.                                             #
###################################################################################

num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)

dx_num = eval_numerical_gradient(lambda x: layers.svm_loss(x, y)[0], x, verbose=False)
loss, dx = layers.svm_loss(x, y)

# Test svm_loss function. Loss should be around 9 and dx error should be 1e-9

print 'Testing svm_loss:'
print 'loss: (should be around 9): ', loss
print 'dx error (should be around 1e-9): ', rel_error(dx_num, dx)

dx_num = eval_numerical_gradient(lambda x: layers.softmax_loss(x, y)[0], x, verbose=False)
loss, dx = layers.softmax_loss(x, y)

# Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8

print '\nTesting softmax_loss:'
print 'loss (should be around 2.3): ', loss
print 'dx error (should be around 1e-8): ', rel_error(dx_num, dx)

# Problem 3.1.5
###################################################################################
#   Two-layer network                                                             #
###################################################################################
#   Now that you have implemented modular versions of the necessary               #
#   layers, you will reimplement a two layer fully connected network using        #
#   these modular implementations.  Open the file fc_net.py and complete          #
#   the implementation of the TwoLayerNet class. This class will serve as         #
#   a model for the other networks you will implement in this assignment,         #
#   so read through it to make sure you understand the API. Then run              #
#   the code below to test your implementation.                                   #
###################################################################################

# input -- hidden -- output
# d units in input layer, h units in hidden layer, C units in output layer

m, d, h, C = 3, 5, 50, 7
X = np.random.randn(m, d)
y = np.random.randint(C, size=m)

std = 1e-2
model = fc_net.TwoLayerNet(input_dim=d, hidden_dim=h, num_classes=C, weight_scale=std)

if model.params != {}:
  print 'Testing initialization ... '
  theta1_std = abs(model.params['theta1'].std() - std)
  theta1_0 = model.params['theta1_0']
  theta2_std = abs(model.params['theta2'].std() - std)
  theta2_0 = model.params['theta2_0']
  assert theta1_std < std / 10, 'First layer weights do not seem right'
  assert np.all(theta1_0 == 0), 'First layer biases do not seem right'
  assert theta2_std < std / 10, 'Second layer weights do not seem right'
  assert np.all(theta2_0 == 0), 'Second layer biases do not seem right'

  print 'Testing test-time forward pass ... '
  model.params['theta1'] = np.linspace(-0.7, 0.3, num=d*h).reshape(d, h)
  model.params['theta1_0'] = np.linspace(-0.1, 0.9, num=h)
  model.params['theta2'] = np.linspace(-0.3, 0.4, num=h*C).reshape(h, C)
  model.params['theta2_0'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=m*d).reshape(d, m).T
  output = model.loss(X)
  correct_output = np.asarray(
    [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
     [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
     [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
  output_diff = np.abs(output - correct_output).sum()
  assert output_diff < 1e-6, 'Problem with test-time forward pass'

print 'Testing training loss (no regularization)'
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
if loss > 0:
  correct_loss = 3.4702243556
  assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

model.reg = 1.0
loss, grads = model.loss(X, y)
if loss > 0:
  correct_loss = 26.5948426952
  assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

for reg in [0.0, 0.7]:
  print 'Running numeric gradient check with reg = ', reg
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))

###################################################################################
#   Solver                                                                        #
###################################################################################
# Following a more modular design, we have split the logic for training models    #
# into a separate class.  Open the file solver.py and read through it to          #
# familiarize yourself with the API. After doing so, use a Solver                 #
# instance to train a TwoLayerNet that achieves at least 50% accuracy on          #
# the validation set.                                                             #
###################################################################################

model = fc_net.TwoLayerNet()
sgd_solver = None

# Problem 3.1.6
###################################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves at least       #
# 50% accuracy on the validation set.                                             #
###################################################################################

pass
##################################################################################
#                             END OF YOUR CODE                                   #
##################################################################################

# Run this code to visualize training loss and train / val accuracy
if sgd_solver:
  plt.subplot(2, 1, 1)
  plt.title('Training loss')
  plt.plot(sgd_solver.loss_history, '-o')
  plt.xlabel('Iteration')

  plt.subplot(2, 1, 2)
  plt.title('Accuracy')
  plt.plot(sgd_solver.train_acc_history, '-o', label='train')
  plt.plot(sgd_solver.val_acc_history, '-o', label='val')
  plt.plot([0.5] * len(sgd_solver.val_acc_history), 'k--')
  plt.xlabel('Epoch')
  plt.legend(loc='lower right')
  plt.gcf().set_size_inches(15, 12)
  plt.show()


# Problem 3.1.7
###################################################################################
#   Multilayer network                                                            #
###################################################################################
#   Next you will implement a fully-connected network with an arbitrary           #
#   number of hidden layers.  Read through the FullyConnectedNet class in         #
#   the file fc_net.py.  Implement the initialization,                            #
#   the forward pass, and the backward pass. For the moment don't worry           #
#   about implementing dropout; we will add that feature next.                    #
###################################################################################

###################################################################################
#   Initial loss and gradient check                                               #
#   As a sanity check, run the following to check the initial loss and to         #
#   gradient check the network both with and without regularization. Do           #
#   the initial losses seem reasonable?  For gradient checking, you should        #
#   expect to see errors around 1e-6 or less.                                     #
###################################################################################

m, d, h1, h2, C = 2, 15, 20, 30, 10

X = np.random.randn(m, d)
y = np.random.randint(C, size=(m,))

for reg in [0, 3.14]:
  print 'Running check with reg = ', reg
  model = fc_net.FullyConnectedNet([h1, h2], input_dim=d, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print 'Initial loss: ', loss

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))

# Problem 3.1.8
###################################################################################
#   TODO: Use a three-layer Net to overfit 50 training examples.                  #
#   As another sanity check, make sure you can overfit a small dataset of         #
#   50 images. First we will try a three-layer network with 100 units in          #
#   each hidden layer. You will need to tweak the learning rate and               #
#   initialization scale, but you should be able to overfit and achieve           #
#   100% training accuracy within 20 epochs.                                      #
###################################################################################

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

# TODO: tweak the values of these two parameters

weight_scale = 1e-2
learning_rate = 1e-4

model = fc_net.FullyConnectedNet([100, 100],
              weight_scale=weight_scale, dtype=np.float64)

if model.params != {}:
  sgd_solver = solver.Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
  sgd_solver.train()

  plt.plot(sgd_solver.loss_history, '-o')
  plt.title('Training loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Training loss')
  plt.show()

# Problem 3.1.9
###################################################################################
# TODO: Use a five-layer Net to overfit 50 training examples.                     #
# Now try to use a five-layer network with 100 units on each layer to             #
# overfit 50 training examples. Again you will have to adjust the                 #
# learning rate and weight initialization, but you should be able to              #
# achieve 100% training accuracy within 20 epochs.                                #
###################################################################################

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}


learning_rate = 1e-3
weight_scale = 1e-5


model = fc_net.FullyConnectedNet([100, 100, 100, 100],
                weight_scale=weight_scale, dtype=np.float64)

if model.params != {}:
  sgd_solver = solver.Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
  sgd_solver.train()

  plt.plot(sgd_solver.loss_history, '-o')
  plt.title('Training loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Training loss')
  plt.show()


###################################################################################
#  Update rules                                                                   #
#  So far we have used vanilla stochastic gradient descent (SGD) as our           #
#  update rule. More sophisticated update rules can make it easier to             #
#  train deep networks. We will implement a few of the most commonly used         #
#  update rules and compare them to vanilla SGD.                                  #
###################################################################################

# Problem 3.1.10
###################################################################################
#  SGD+Momentum                                                                   #
#  Stochastic gradient descent with momentum is a widely used update rule         #
#  that tends to make deep networks converge faster than vanilla                  #
#  stochastic gradient descent.  Open the file optim.py and read                  #
#  the documentation at the top of the file to make sure you understand           #
#  the API. Implement the SGD+momentum update rule in the function                #
#  sgd_momentum and run the following to check your implementation. You           #
#  should see errors less than 1e-8.                                              #
###################################################################################

from optim import sgd_momentum

m, d = 4, 5
theta = np.linspace(-0.4, 0.6, num=m*d).reshape(m, d)
dtheta = np.linspace(-0.6, 0.4, num=m*d).reshape(m, d)
v = np.linspace(0.6, 0.9, num=m*d).reshape(m, d)

config = {'learning_rate': 1e-3, 'velocity': v}
next_theta, _ = sgd_momentum(theta, dtheta, config=config)

expected_next_theta = np.asarray([
  [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
  [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
  [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
  [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
expected_velocity = np.asarray([
  [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
  [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
  [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
  [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

if next_theta:
  print 'next_theta error: ', rel_error(next_theta, expected_next_theta)
  print 'velocity error: ', rel_error(expected_velocity, config['velocity'])


###################################################################################
#  Compare SGD and SGD-momentum                                                   #
#  Run the following to train a six-layer network                                 #
#  with both SGD and SGD+momentum. You should see the SGD+momentum update         #
#  rule converge faster.                                                          #
###################################################################################

num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}

for update_rule in ['sgd', 'sgd_momentum']:
  print 'running with ', update_rule
  model = fc_net.FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

  if model.params != {}:
    asolver = solver.Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': 1e-2,
                  },
                  verbose=True)
    solvers[update_rule] = asolver
    asolver.train()
    print

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

for update_rule, asolver in solvers.iteritems():
  plt.subplot(3, 1, 1)
  plt.plot(asolver.loss_history, '-o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(asolver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(asolver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()

# Problem 3.1.11
###################################################################################
#  RMSProp and Adam are update rules that set per-parameter                       #
#  learning rates by using a running average of the second moments of             #
#  gradients.  In the file optim.py, implement the RMSProp update                 #
#  rule in the rmsprop function.                                                  #
#  Test RMSProp implementation; you should see errors less than 1e-7              #
###################################################################################

from optim import rmsprop

m, d = 4, 5

theta = np.linspace(-0.4, 0.6, num=m*d).reshape(m, d)
dtheta = np.linspace(-0.6, 0.4, num=m*d).reshape(m, d)
cache = np.linspace(0.6, 0.9, num=m*d).reshape(m, d)

config = {'learning_rate': 1e-2, 'cache': cache}
next_theta, _ = rmsprop(theta, dtheta, config=config)

expected_next_theta = np.asarray([
  [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
  [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
  [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
  [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
expected_cache = np.asarray([
  [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
  [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
  [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
  [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])

if next_theta:
  print 'next_theta error: ', rel_error(expected_next_theta, next_theta)
  print 'cache error: ', rel_error(expected_cache, config['cache'])

###################################################################################
# Test Adam implementation; you should see errors around 1e-7 or less             #
###################################################################################
from optim import adam

m, d = 4, 5
theta = np.linspace(-0.4, 0.6, num=m*d).reshape(m, d)
dtheta = np.linspace(-0.6, 0.4, num=m*d).reshape(m, d)

mo = np.linspace(0.6, 0.9, num=m*d).reshape(m, d)
v = np.linspace(0.7, 0.5, num=m*d).reshape(m,d)


config = {'learning_rate': 1e-2, 'm': mo, 'v': v, 't': 5}
next_theta, _ = adam(theta, dtheta, config=config)

expected_next_theta = np.asarray([
  [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
  [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
  [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
  [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
expected_v = np.asarray([
  [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
  [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
  [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
  [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
expected_m = np.asarray([
  [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
  [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
  [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
  [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])

print 'next_theta error: ', rel_error(expected_next_theta, next_theta)
print 'v error: ', rel_error(expected_v, config['v'])
print 'm error: ', rel_error(expected_m, config['m'])


###################################################################################
# Once you have debugged your RMSProp, run the                                    #
# following to train a pair of deep networks using these new update rules.        #
# the Adam rule has been implemented for you in optim.py                          #
###################################################################################


learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
for update_rule in ['adam', 'rmsprop']:
  print 'running with ', update_rule
  model = fc_net.FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
  if model.params != {}:
    asolver = solver.Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': learning_rates[update_rule]
                  },
                  verbose=True)
    solvers[update_rule] = asolver
    asolver.train()
    print


    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

for update_rule, asolver in solvers.iteritems():
  plt.subplot(3, 1, 1)
  plt.plot(asolver.loss_history, '-o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(asolver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(asolver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()


# Problem 3.1.12
###################################################################################
# Tune your hyperparameters                                                       #
#                                                                                 #
# Look to see if there is a gap between                                           #
# the training and validation accuracy, suggesting that the model we              #
# used is overfitting, and that we should decrease its size. On the               #
# other hand, with a small model we would expect to see more                      #
# underfitting, which would manifest itself as no gap between                     #
# the training and validation accuracy. The learned filters are noisy,            #
# indicating there is room for improvement                                        #
#                                                                                 #
# Tuning the                                                                      #
# hyperparameters and developing intuition for how they affect the final          #
# performance is a large part of using Neural Networks, so we want you            #
# to get a lot of practice. Below, you should experiment with different           #
# values of the various hyperparameters, including hidden layer sizes,            #
# number of hidden layers, learning rate, numer of training epochs, and           #
# regularization strength. You might also consider varying the learning rule      #
# RMSProp and Adam are good choices.                                              #
#
#                                                                                 #
# You should be aim to achieve a classification                                   #
# accuracy of greater than 50% on the validation set. Our best network            #
# gets over 56% on the validation set.  Experiment: You goal in this              #
# exercise is to get as good of a result on CIFAR-10 as you can, with a           #
# fully-connected Neural Network.                                                 #
###################################################################################


from vis_utils import show_net_weights

# TODO: 
# set up a model, train it and then visualize the first level weights
# you will need to play wih these parameters to get > 50% on validation set

# model = fc_net.FullyConnectedNet()

# asolver = solver.Solver(model, data,
#                  num_epochs=5, batch_size=100,
#                  update_rule='adam',
#                  optim_config={
#                    'learning_rate': 1e-3,
#                  },
#                  verbose=True)
# asolver.train()
# show_net_weights(model)
