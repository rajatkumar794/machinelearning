# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:01:57 2019

@author: rajat
"""
import numpy as np # linear algebra
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tfc
tfc.disable_v2_behavior() 
from batch import random_mini_batches

def forward_propagation_for_predict(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
    

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tfc.placeholder("float", [4096, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tfc.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

X = np.load("Xfixed.npy")
Y = np.load("Yfixed.npy")
X = X.reshape(X.shape[0],-1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def create_placeholders(n_x, n_y):
    X = tfc.placeholder(tf.float32, shape=[n_x,None], name='X')
    Y = tfc.placeholder(tf.float32, shape=[n_y,None], name='Y')
    return X, Y

def initialize_parameters():
    W1 = tfc.get_variable('W1', [30,4096], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tfc.get_variable('b1', [30,1], initializer = tfc.zeros_initializer())
    W2 = tfc.get_variable('W2', [15,30], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tfc.get_variable('b2', [15,1], initializer = tfc.zeros_initializer())
    W3 = tfc.get_variable('W3', [10,15], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tfc.get_variable('b3', [10,1], initializer = tfc.zeros_initializer())
    parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)
    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)
    (nx, m) = np.shape(X_train)
    ny = np.shape(Y_train)[0]
    costs = []
    x, y = create_placeholders(nx, ny)
    parameters = initialize_parameters()
    Z3 = forward_propagation(x, parameters)
    cost = compute_cost(Z3, y)
    optimizer = tfc.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tfc.global_variables_initializer()
    with tfc.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                (batch_X, batch_Y) = minibatch
                _,batch_cost = sess.run([optimizer, cost], feed_dict={x:batch_X, y:batch_Y})
                epoch_cost += batch_cost / minibatch_size
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(costs)
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}, session=sess))
        print ("Test Accuracy:", accuracy.eval({x: X_test, y: Y_test}, session=sess))
    return parameters

parameters = model(X_train, Y_train, X_test, Y_test)

import imageio as imgr
from skimage import color
fname = "sign1.jpg"

img = np.array(imgr.imread(fname))
img = tf.image.resize(img, (64, 64))
with tfc.Session() as sess:
    img=sess.run(img)
    
img_gray = color.rgb2gray(img)
image= img_gray.reshape(64*64,1)
my_image_prediction = predict(image, parameters)
plt.imshow(img_gray)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

import pickle as pkl
with open('parameter.pkl', 'wb') as wf:
    pkl.dump(parameters, wf, pkl.HIGHEST_PROTOCOL)