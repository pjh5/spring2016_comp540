from sklearn import linear_model
import numpy as np

class one_vs_allLogisticRegressor:

    def __init__(self,labels):
        self.theta = None
        self.labels = labels

    def train(self,X,y,reg,penalty):
        
        """
        Use sklearn LinearRegression for training K classifiers in one-vs-rest mode
        with X,y and select the right solver depending on penalty
        If penalty is l2, choose l_bfgs, if penalty is l1, choose liblinear
        
        X = m X d array of training data. Assumes X has an intercept column
        y = 1 dimensional vector of length m (with K labels)
        reg = regularization strength
        penalty = 'l1' or 'l2'
        Returns coefficents for K classifiers: a matrix with K rows and d columns
           - one theta of length d for each class
       """
        
        m,dim = X.shape
        theta_opt = np.zeros((len(self.labels),dim))

        ###########################################################################
        # Train the classifier in 1 vs rest mode .                                #
        # For each label do:                                                      #
        #   make label the positive class, and the other labels the neg class     #
        #   assemble the appropriate X and y with the new label scheme            #
        #   train a logistic classifier using sklearn's LogisticRegression        #
        #   store the coefficients in a row of theta_opt                          #
        # TODO: 7-9 lines of code expected                                        #
        ###########################################################################



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        self.theta = theta_opt
        return theta_opt

    def predict(self,X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: m x d array of training data. 

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
          array of length m, and each element is a class label from one of the
          set of labels -- the one with the highest probability
        """

        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 2 lines of code expected                                          #
        ###########################################################################



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

