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
from layer_utils import conv_relu_pool_forward, conv_relu_pool_backward
from layer_utils import conv_relu_forward, conv_relu_backward
import layers
import sys

data = get_CIFAR10_data()

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

import cnn3
model = cnn3.ConvNet(weight_scale=0.03, num_filters=[96,256,384], hidden_dim=[1000, 1000], reg=0.0001, load=False)
"""
if model.params != {}:
  asolver = solver.Solver(model, small_data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 5e-5,
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
"""
sys.stdout.flush()
###################################################################################
# Train the net                                                                   #
#                                                                                 #
# By training the three-layer convolutional network for one epoch, you            #
# should achieve greater than 40% accuracy on the training set:                   #
###################################################################################

#model = cnn2.ConvNet(weight_scale=0.05, num_filters=[32,32,32], hidden_dim=500, reg=0.001)

if model.params != {}:
  asolver = solver.Solver(model, data,
                num_epochs=5, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': 5e-5,
                },
                verbose=True, print_every=50)
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
