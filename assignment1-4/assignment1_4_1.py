import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 2层网络参数初始化
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x - size of the input layer
    n_h - size of the hidden layer
    n_y - size of the output layer

    Return:
    parameters - python dictionary containing your parameters:
                 W1 - weight matrix of shape (n_h, n_x)
                 b1 - bias vector of shape (n_h, 1)
                 W2 - weight matrix of shape (n_y, n_h)
                 b2 - bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}

    return parameters

'''
# function test
parameters = initialize_parameters(2, 2, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

# L层网络参数初始化
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims - python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters: - python dictionary containing your parameters
                  Wl - weight matrix of shape (layer_dims[l], layer_dims[l-1])
                  bl - bias matrix of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #* 0.01 涉及超参数初始化
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

'''
# function test
parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters['W1']))
print("b1 = " + str(parameters['b1']))
print("W2 = " + str(parameters['W2']))
print("b2 = " + str(parameters['b2']))
'''

# 前向传播（单层）的linear部分
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward proppagation

    Arguments:
    A - activations from previous layer (or input data): (size of previous layers, number of examples)
    W - 
    b -

    Returns:
    Z - the input of the activation function, also called pre-activation parameter
    cache - a python dictionary containing "A", "W" and "b"; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

'''
# function test
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))'''

# 前向传播(单层)
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev - activations from previous layer (or input data): (size of previous layers, number of examples)
    W -
    b -
    activation - the activation to be used in this layer, stored as a text string:"sigmoid" or "relu"

    Returns:
    A - the output of the activation function, also called the post-activation value
    cache - a python dictionary containing "linear_cache" and "activation_cache": stroed for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

'''
# function test
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))
'''

#前向传播
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X - data, numpy array of shape (input size, number of examples)
    parameters - output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - list of caches containing:
             every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
             the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters)//2      # number of layers in the neural network(这里已经去掉了输入层，因为输入层不计入神经网络的层数)

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

'''
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
'''

# 计算成本
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL - probability vector corresponding to your label predictions, shape(1, number of examples)
    Y - true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape(1, number of examples)

    Returns:
    cost - cross-entropy cost 交叉熵
    """

    m = Y.shape[1]

    cost = -(np.dot(Y, np.log(AL.T)) + np.dot(1-Y, np.log(1-AL.T)))/m

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

'''
# function test
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
'''

# 反向传播的linear部分（单层）
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer l

    Arguments:
    dZ - Gradient of the cost with respect to the linear output (of current layer l)
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev - Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW - Gradient of the cost with respect to W (current layer l), same shape as W
    db - Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

'''
# function test
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print('dA_prev = ' + str(dA_prev))
print('dW = ' + str(dW))
print('db = ' + str(db))
'''

# 反向传播（单层）
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA - post-activation gradient for current layer l
    cache - tuple of values(linear_cache, activation_cache) we store for computing ackward propagation efficiently
    activation - the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev - Gradient of the cost with respect to the activation (of the previous layer l-1)
    dW - 
    db-
    """
    
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

'''
# function test
AL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print("sigmoid")
print("dA_prev = " + str(dA_prev))
print('dW = ' + str(dW))
print('db = ' + str(db) + '\n')
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print("relu")
print("dA_prev = " + str(dA_prev))
print('dW = ' + str(dW))
print('db = ' + str(db))
'''

# 反向传播
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR->SIGMOID group

    Arguments:
    AL - probability vector, output of the forward propagation (L_model_forward)
    Y - true "label" vector (containing 0 if non-cat, 1 if cat)
    caches - list of caches containing:
             every cache of linear_activation_forward with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
             the cache of linear_activation_foward with "sigmoid" (it's caches[L-1])

    Returns:
    grads - A dictionary with the gradients
            grads["dA" + str(l)] = ...
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL)) # 这里的dAL相当于成本函数对网络输出AL的偏微分

    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid") # 这里的dAL相当于反向传播第L层的输出

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, activation = "relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        

    return grads

'''
# function test
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dA1 = " + str(grads["dA1"]))
'''

# 参数更新
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient desecent

    Arguments:
    parameters -
    grads -

    Returns:
    parameters -
    """

    L = len(parameters)//2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

'''
# function test
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''
