#!/usr/bin/env conda run --no-capture-output -n vision python

# if you aren't using conda run -n, you need to manually activate the environment ( but disabeling output buffering might not work)

import numpy as np
import matplotlib.pyplot as plt
# from mnist import data # somehow accidentally deleted the file
from images import mnist
from data_prep import one_hot
from activations import relu, sigmoid, swish


class mlp:
    def __init__(self, *, X, Y, X_test, Y_test, layers, activation, iterations, lr, BS, backward, LR_range_test):
        self.lr = lr
        self.lr_list = [self.lr]
        self.LR_range_test = LR_range_test
        self.BS = BS
        self.layers = layers
        self.activation = activation
        self.iterations = iterations
        self.backward = backward
        self.grad_w = []

        self.w = []
        self.b = []
        # normalized input data
        X = (X - np.mean(X[:self.BS])) / np.std(X[:self.BS])
        self.f = [[]]
        self.loss = []
        self.test_loss = []

        self.train(X, Y, X_test, Y_test)

    def evaluation(self, X, Y, test=False):
        if test == True:
            self.f[0] = X
            for j in range(len(self.layers)):
                self.forward(X, j)
            avg_BS_loss = np.mean((self.f[-1] - Y) ** 2)
            self.test_loss.append(
                (np.argmax(self.f[-1], axis=1) == np.argmax(Y, axis=1)).mean())
        else:
            avg_BS_loss = (
                np.argmax(self.f[- 1], axis=1) == np.argmax(Y, axis=1)).mean()
            exp_scores = np.exp(
                self.f[-1] - np.max(self.f[-1], axis=1).reshape(self.BS, 1))
            # subtracting max(f) doesn't change the result, it solely to improve to improve the numerical stability

            softmax = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
            cross_entropy = Y * -np.log(softmax)
            avg_BS_loss = np.mean(cross_entropy)
            self.loss.append(np.mean(avg_BS_loss))

    def train(self, X, Y, X_test, Y_test):
        self.d = []
        self.param_init(X[1].shape, self.backward)
        for i in range(self.iterations):
            if self.LR_range_test:
                self.lr = self.lr - 0.001
                self.lr_list.append(self.lr)
            samp = np.random.randint(0, 5000, size=(self.BS))
            self.f[0] = X[samp]
            for j, _ in enumerate(self.layers):
                self.forward(X[samp], j)
            for j, _ in enumerate(self.layers):
                self.backward_pass(X[samp], Y[samp], j)
            self.evaluation(X, Y[samp])
            # if i % 1 == 0:
            #     self.evaluation(X_test.reshape(-1, 784)
            #                     [samp], Y_test[samp], True)
            if i % 100 == 0:
                self.evaluation(X_test.reshape(-1, 784)
                                [samp], Y_test[samp], True)
                print('model output', np.argmax(self.f[len(self.f) - 1][:10], axis=1))
                print('label       ', np.argmax(Y_test[samp][:10], axis=1))

        print("model output on test set", np.argmax(self.f[len(self.f) - 1], axis=1))
        print("labels", np.argmax(Y[samp], axis=1))
        print(f'last loss {self.loss[-1]}')
        plt.plot(self.loss)
        plt.show()
        print(f"test acc. after training {self.test_loss[-1]}")
        print(f"test acc. before training {self.test_loss[0]}")
        plt.plot(self.test_loss)
        plt.show()

    def param_init(self, input_shape, backward):
        for i in range(len(self.layers)):
            scale_w = 0.001
            scale_b = 0.005
            self.w.append(np.random.uniform(0, 1, size=(
                self.layers[i][0], self.layers[i][1])) * scale_w)
            self.b.append(np.random.random((1, self.layers[i][1])) * scale_b)
            self.f.append(np.empty(()))
            if backward:
                self.d.append(np.empty(()))
                self.grad_w.append(np.empty(()))

    def forward(self, X, i):
        self.f[i + 1] = self.activation[i]((self.f[i] @ self.w[i]) + self.b[i])

    def backward_2(self, X, Y, i):
        mse_d = 2*(self.f[- 1] - Y)
        self.d[0] = mse_d * sigmoid(self.f[2], True)/self.BS
        grad_w1 = self.d[0].T @ self.f[1]
        grad_b1 = np.mean(self.d[0], axis=0, keepdims=True)

        self.d[1] = self.d[0] @ self.w[1].T * sigmoid(self.f[1], True)
        grad_w0 = self.d[1].T @ self.f[0]
        grad_b0 = np.mean(self.d[1], axis=0, keepdims=True)
        self.w[1] = self.w[1] + self.lr * grad_w1.T
        self.b[1] = self.b[1] + self.lr * grad_b1
        self.w[0] = self.w[0] + self.lr * grad_w0.T
        self.b[0] = self.b[0] + self.lr * grad_b0

    def backward_pass(self, X, Y, i):
        if i == 0:
            exp_scores = np.exp(
                self.f[- 1] - np.max(self.f[-1], axis=1).reshape(self.BS, 1))
            # subtracting max(f) doesn't change the result, it solely to improve to improve the numerical stability

            softmax = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
            d_cross_entropy = np.copy(softmax)
            d_cross_entropy[np.arange(np.argmax(Y, axis=1).size), np.argmax(
                Y, axis=1)] = softmax[np.arange(np.argmax(Y, axis=1).size), np.argmax(Y, axis=1)] - 1

            # depending on your loss function
            self.d[i] = d_cross_entropy * \
                self.activation[- 1](self.f[- 1], True)/self.BS
            # dividing by the batch size (not necessary to do for each gradient as this gets backpropagated)

            self.grad_w[i] = self.d[i].T @ self.f[- 2]
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)
            self.w[- 1] = self.w[- 1] + self.lr * self.grad_w[i].T
            self.b[- 1] = self.b[- 1] + self.lr * grad_b

        else:
            self.d[i] = self.d[i - 1] @ (self.w[- i].T - self.lr * self.grad_w[i - 1]
                                         ) * self.activation[- i](self.f[- i - 1], True)

            self.grad_w[i] = self.d[i].T @ self.f[- 2 - i]
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)

            self.w[- 1 - i] = self.w[- 1 - i] + self.lr * self.grad_w[i].T
            self.b[- 1 - i] = self.b[- 1 - i] + self.lr * grad_b


X_train, Y_train, X_test, Y_test = mnist()
Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

mlp(
    X=X_train.reshape(-1, 784),
    Y=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    layers=[[784, 128], [128, 128], [128, 10]],
    activation=[relu, relu, relu],
    iterations=5000,
    lr=-0.5,
    BS=256,
    backward=True,
    LR_range_test=False)
