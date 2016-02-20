import random
import numpy as np
import matplotlib.pyplot as plt
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time

# Get the CIFAR-10 data broken up into train, validation and test sets

X_train, y_train, X_val, y_val, X_test, y_test = utils.get_CIFAR10_data()

# First implement the naive softmax loss function with nested loops.
# Open the file softmax.py and implement the
# softmax_loss_naive function.

# Generate a random softmax theta matrix and use it to compute the loss.

theta = np.random.randn(3073,10) * 0.0001
loss, grad = softmax_loss_naive(theta, X_train, y_train, 0.0)

# Loss should be something close to - log(0.1)

print 'loss:', loss, ' should be close to ', - np.log(0.1)

# Use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient. (within 1e-7)

from gradient_check import grad_check_sparse
f = lambda th: softmax_loss_naive(th, X_train, y_train, 0.0)[0]
grad_numerical = grad_check_sparse(f, theta, grad, 10)

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.

tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)


# We use the Frobenius norm to compare the two versions
# of the gradient.

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
print 'Gradient difference: %f' % grad_difference

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set and the test set.

results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
regularization_strengths = [5e4, 1e5, 5e5, 1e8]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# Save the best trained softmax classifer in best_softmax.                     #
# Hint: about 10 lines of code expected
################################################################################



################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate the best softmax classifier on test set

if best_softmax:
  y_test_pred = best_softmax.predict(X_test)
  test_accuracy = np.mean(y_test == y_test_pred)
  print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

  # Visualize the learned weights for each class

  theta = best_softmax.theta[1:,:].T # strip out the bias term
  theta = theta.reshape(10, 32, 32, 3)

  theta_min, theta_max = np.min(theta), np.max(theta)

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for i in xrange(10):
    plt.subplot(2, 5, i + 1)
  
    # Rescale the weights to be between 0 and 255
    thetaimg = 255.0 * (theta[i].squeeze() - theta_min) / (theta_max - theta_min)
    plt.imshow(thetaimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])


  plt.savefig('cifar_theta.pdf')
  plt.close()
