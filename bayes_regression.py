import sys
import os

import tensorflow as tf
import edward as ed
from edward.models import Normal

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('J:\\Analytics\\Shared\\98 Analytics Team\\24 Ton\\05 Python\\UsefulCode'))
import utils

class BayesianRegression:

    def __init__(self, in_dim=1, n_classes=2):
        """
        Bayesian Logistic regression based on Edward lib (http://edwardlib.org).

        y = W * x + b

        :param in_dim:
        :param n_classes:
        """

        self.in_dim = in_dim
        self.n_classes = n_classes

        self.X = tf.placeholder(tf.float32, [None, self.in_dim])
        self.W = Normal(loc=tf.zeros([self.in_dim, self.n_classes]), scale=tf.ones([self.in_dim, self.n_classes]))
        self.b = Normal(loc=tf.zeros(self.n_classes), scale=tf.ones(self.n_classes))

        h = tf.matmul(self.X, self.W) + self.b
        self.y = Normal(loc=tf.sigmoid(-h), scale=0.1)

        self.qW = Normal(loc=tf.get_variable("qW/loc", [self.in_dim, self.n_classes]), scale=tf.nn.softplus(tf.get_variable("qW/scale", [self.in_dim, self.n_classes])))
        self.qb = Normal(loc=tf.get_variable("qb/loc", [self.n_classes]), scale=tf.nn.softplus(tf.get_variable("qb/scale", [self.n_classes])))

    def infer(self, X, y, n_samples=5, n_iter=250):

        inference = ed.KLqp({self.W: self.qW,
                             self.b: self.qb,}, data={self.y: y, self.X: X})

        inference.run(n_samples=n_samples, n_iter=n_iter)

    def predict(self, X):

        self.qW_mean = self.qW.mean().eval()
        self.qb_mean = self.qb.mean().eval()

        h = tf.matmul(X, self.qW_mean) + self.qb_mean

        return tf.sigmoid(-h).eval()

    def sample_boudary(self, X):

        qW= self.qW.eval()
        qb = self.qb.eval()

        w = - qW[0][0] / qW[1][0]
        b = (0.5 - qb[0]) / qW[0][0]

        return w, b

    def predict_std(self, X):
        self.qW_stddev = self.qW.stddev().eval()
        self.qb_stddev = self.qb.stddev().eval()

        h = tf.matmul(X, self.qW_stddev) + self.qb_stddev

        return tf.sigmoid(-h).eval()

    def get_coef(self):
        return self.qW.mean().eval().T[0]


if __name__ == "__main__":

    m, mt = ([4,4], [0, 0])
    C = np.array([[3, 0], [0, 3]])
    N = 1000
    X1 = np.random.multivariate_normal(m, C, N)
    X2 = np.random.multivariate_normal(mt, C, N)
    X = np.array(np.vstack((X1, X2)), dtype=np.float32)

    y1 = np.concatenate((np.zeros(N), np.ones(N)))
    y2 = np.concatenate((np.ones(N), np.zeros(N)))
    y = np.array(np.vstack((y1, y2)).T, dtype=np.float32)

    bnn = BayesianRegression(in_dim=2, n_classes=2)
    bnn.infer(X, y, n_samples=10, n_iter=500)

    plt.plot(X1.T[0], X1.T[1], 'b.', label='Class 0')
    plt.plot(X2.T[0], X2.T[1], 'r.', label='Class 1')
    for _ in range(10):
        w, b = bnn.sample_boudary(X)
        plt.plot([7.5, -5] , [w * 7.5 + b, w * -5 + b])

    plt.legend()
    plt.show()

    f = bnn.predict(X)
    print("Class 0")
    utils.print_results(f.T[0], y1)

    print("Class 1")
    utils.print_results(f.T[1], y2)


    # plt.fill_between(x_train.T[0], y_mean - 3 * y_std, y_mean + 3 * y_std, facecolor='blue', alpha=0.3)
    # plt.fill_between(x_train.T[0], y_mean - y_std, y_mean + y_std, facecolor='blue', alpha=0.3)
    # plt.plot(x_train, y_mean)
    # plt.plot(x_train, y_train, '.')
    # plt.show()

