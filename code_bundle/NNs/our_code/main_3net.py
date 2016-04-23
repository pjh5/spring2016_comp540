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

import numpy as np
import  utils
import fc_net 
import solver

###################################################################################
#   Load the (preprocessed) CIFAR10 data.                                         #
###################################################################################

data = utils.get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape


model = fc_net.FullyConnectedNet([500,500],
             weight_scale=1e-3, dtype=np.float64, reg=0.01)

asolver = solver.Solver(model, data,
                 num_epochs=20, batch_size=200,
                 update_rule='adam',
                 optim_config={
                   'learning_rate': 2e-4,
                 },
                 verbose=True)
asolver.train()
