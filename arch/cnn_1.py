#!/usr/bin/env conda run --no-capture-output -n vision python

import numpy as np
from images import mnist
from data_prep import one_hot
from activations import relu, sigmoid, swish
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

image_s = Image.open('path_to_image')
image_s = ImageOps.grayscale(image_s)
image_s = np.array(image_s)
image_s = image_s[0:600, 200:800] /image_s.max()
print(image_s.shape)
label = Image.open('label.jpeg')
label = np.array(label).astype(np.float64)

X_train, Y_train, X_test, Y_test = mnist()

feature_1 = (X_train.reshape(-1, 784)[0].reshape(28, 28) - X_train.mean())
feature_1 = image_s


N = 600
F = 3
stride = 1
map_1 = np.empty((int((N-F)/stride+1), int((N-F)/stride+1)))

#forward
def forward(feature_1, filter):
    map_1 = np.zeros((int((N-F)/stride + 1), int((N-F)/stride + 1)))
    for i in range(int((N-F)/stride + 1) - F + 1):
        for j in range(int((N-F)/stride + 1) -F + 1):
            split = feature_1[i*stride:i*stride+F, j*stride:j*stride+F]
            map_1[i, j] = np.sum(split  * filter)
    return map_1
#im = Image.fromarray((map_1 * 255).astype(np.uint8))
#im.save('label.jpeg')



# backprop
def backward(filter, map_1, label):
    N = 600
    loss = 2*(map_1 - label).mean()
    print(loss)

    grad_1 = np.zeros((F, F))
    for i in range(int((N-F)/stride + 1) - F):
        for j in range(int((N-F)/stride + 1) - F):
            grad_1[:F, :F] += loss * feature_1[i+stride:F+i+stride, j+stride:F+j+stride]

    grad_1 = grad_1
    filter += -8e-7* grad_1
    return filter

def first():
    feature_1 = image_s
    filter = np.array([[0,-2,1],[-1,-1,1],[1,2,1]]).astype(np.float64)
    filter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).astype(np.float64)
    #filter = np.ones((3,3)).astype(np.float64)
    label = forward(feature_1, filter)
    #plt.imshow(label)
    #plt.show()
    filter = np.random.uniform(1,-1, size=(3,3)) 
    """
    for i in range(40):
        map_1 = forward(feature_1, filter)
        filter = backward(filter, map_1, label)
    plt.imshow(map_1)
    plt.show()
    """
    return label
    
label = first()
print(label.mean())
filter = np.array([[0,-2,1],[-1,-1,1],[1,2,1]]).astype(np.float64)
filter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).astype(np.float64)

#filter = np.ones((3,3)).astype(np.float64)
#x = np.convolve(filter.flatten(), feature_1.flatten(), 'same')

#filter = np.rot90(filter, k=1, axes=(0,1)).copy()
#np_x = np.convolve(feature_1.flatten(), filter.flatten(), 'valid')
#np_x = np.einsum('ij,ijkl->kl',feature_1.flatten(),filter.flatten())
#print(np_x.shape)

import torch
input = torch.from_numpy(feature_1).reshape(1, 1, 600, 600)
kernel = torch.from_numpy(filter).reshape(1, 1, 3, 3)
x = torch.nn.functional.conv2d(input, kernel, padding="valid")
torch.set_printoptions(precision=12)
print(x.mean())
#plt.imshow(x.reshape(598,-1))
#plt.show()
