import sys
import os

import tensorflow as tf
import edward as ed
from edward.models import Normal

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('J:\\Analytics\\Shared\\98 Analytics Team\\24 Ton\\05 Python\\UsefulCode'))
import utils

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network based on Edward lib (http://edwardlib.org).

    :param in_dim: Input dimensionality (number of features)
    :param n_classes: Output dimensionality
    :param layers: Tuple of size of each layer, not including output layer.
    :param nonlinearity: Non-linearity function applied to middle layers.
    :param nonlinearity_last: Non-linearity function applied to the output layer.
    """
    def __init__(self, in_dim=1, n_classes=1, layers=(10, 10), nonlinearity=tf.tanh,
                 nonlinearity_last=tf.identity, alpha=1):

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.layers = layers
        self.nonlinearity = nonlinearity
        self.nonlinearity_last = nonlinearity_last
        alpha = float(alpha)

        self.model_infered = False

        # Add output layer to layer tuple
        self.layers = self.layers + (self.n_classes,)
        self.num_layers = len(self.layers)

        # Build neural network
        self.X = tf.placeholder(tf.float32, [None, self.in_dim], name='X')

        l_last = self.in_dim
        self.nn_dict = {'W': [], 'b': [], 'h': [],'qW': [], 'qb': []}
        for i in range(self.num_layers):
            l_current = self.layers[i]

            # Prior probabilities on parameters W and b are Normal dist with mean zero and std alpha
            self.nn_dict['W'].append(Normal(loc=tf.zeros([l_last, l_current]), scale=alpha*tf.ones([l_last, l_current]), name='W_{}'.format(i)))
            self.nn_dict['b'].append(Normal(loc=tf.zeros(l_current), scale=alpha*tf.ones(l_current), name='b_{}'.format(i)))

            # First Layer
            if i == 0:
                self.nn_dict['h'].append(self.nonlinearity(tf.matmul(self.X, self.nn_dict['W'][i]) + self.nn_dict['b'][i], name='h_{}'.format(i)))
            # Output Layer
            elif i == self.num_layers - 1:
                self.nn_dict['h'].append(self.nonlinearity_last(tf.matmul(self.nn_dict['h'][i-1], self.nn_dict['W'][i]) + self.nn_dict['b'][i], name='h_{}'.format(i)))
            # Middle Layers
            else:
                self.nn_dict['h'].append(self.nonlinearity(tf.matmul(self.nn_dict['h'][i-1], self.nn_dict['W'][i]) + self.nn_dict['b'][i], name='h_{}'.format(i)))

            # Posterior probabilities on parameters W and b
            self.nn_dict['qW'].append(Normal(loc=tf.get_variable("qW_{}/loc".format(i), [l_last, l_current]),
                                        scale=alpha*tf.nn.softplus(tf.get_variable("qW_{}/scale".format(i), [l_last, l_current])), name='qW_{}'.format(i)))
            self.nn_dict['qb'].append(Normal(loc=tf.get_variable("qb_{}/loc".format(i), [l_current]),
                                        scale=alpha*tf.nn.softplus(tf.get_variable("qb_{}/scale".format(i), [l_current])), name='qb_{}'.format(i)))

            l_last = l_current

        # Define probability over the outputs as a Normal with mean=outputs and std=alpha
        self.y = Normal(loc=self.nn_dict['h'][-1], scale=alpha, name='y')

    def infer(self, X, y, n_samples=5, n_iter=250, inference_method=ed.KLqp):
        """
        Run inference.

        :param X: Training input data
        :param y: Training output data
        :param n_samples: Number of samples from variational model for calculating stochastic gradients.
        :param n_iter: Number of iterations.
        :param inference_method: Inference function (default to ed.KLqp).
        """

        # Inference_dict maps the prior and posterior probabilities for each W and b
        inference_dict = dict()
        for i in range(self.num_layers):
            inference_dict[self.nn_dict['W'][i]] = self.nn_dict['qW'][i]
            inference_dict[self.nn_dict['b'][i]] = self.nn_dict['qb'][i]

        inference = inference_method(inference_dict, data={self.y: y, self.X: X})

        inference.run(n_samples=n_samples, n_iter=n_iter)
        self.model_infered = True

    def predict(self, X):
        """
        Make a prediction of an X input based on the mean of the probability distribution of the parameters
        'mean[q(params)]'.

        :param X: Data to predict
        :return: Prediction put
        """

        if not self.model_infered:
            raise AttributeError('Model not infered. Please run inference() before predict().')

        for i in range(self.num_layers):

            # Get the mean of 'W' and 'b' for layer i
            qW_mean = self.nn_dict['qW'][i].mean().eval()
            qb_mean = self.nn_dict['qb'][i].mean().eval()

            # First Layer
            if i == 0:
                h = self.nonlinearity(tf.matmul(X, qW_mean) + qb_mean)
            # Last Layer
            elif i == self.num_layers - 1:
                return self.nonlinearity_last(tf.matmul(h, qW_mean) + qb_mean).eval()
            # Middle Layers
            else:
                h = self.nonlinearity(tf.matmul(h, qW_mean) + qb_mean)


if __name__ == "__main__":
    test = 2
    ## Test 1 ##
    if test == 1:
        x_train = np.linspace(-3, 3, num=50)
        y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
        x_train = x_train.astype(np.float32).reshape((50, 1))
        y_train = y_train.astype(np.float32).reshape((50, 1))

        bnn = BayesianNeuralNetwork(in_dim=1, n_classes=1, layers=(10,10,10))
        bnn.infer(x_train, y_train, n_samples=10, n_iter=1000)
        f = bnn.predict(x_train).T[0]

        plt.plot(x_train, y_train, 'x', label='Data')
        plt.plot(x_train, f, 'x', label='Predicted mean')
        plt.legend()
        plt.show()

    ## Test 2 ##
    elif test == 2:
        m, mt = ([4,4], [0, 0])
        C = np.array([[2, 0], [0, 2]])
        n_samples = 1000
        X1 = np.random.multivariate_normal(m, C, n_samples)
        X2 = np.random.multivariate_normal(mt, C, n_samples)
        X = np.array(np.vstack((X1, X2)), dtype=np.float32)

        plt.plot(X1.T[0], X1.T[1], 'b.', label='Class 0')
        plt.plot(X2.T[0], X2.T[1], 'r.', label='Class 1')
        plt.legend()
        plt.show()

        y1 = np.concatenate((np.zeros(n_samples), np.ones(n_samples)))
        y2 = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))
        y = np.array(np.vstack((y1, y2)).T, dtype=np.float32)

        bnn = BayesianNeuralNetwork(in_dim=2, n_classes=2, layers=(20,20))
        bnn.infer(X, y, n_samples=10, n_iter=500)
        f = bnn.predict(X)

        print("Class 0")
        utils.print_results(f.T[0], y1)

        print("Class 1")
        utils.print_results(f.T[1], y2)

