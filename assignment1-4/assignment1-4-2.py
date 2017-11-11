import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import load_data, predict, print_mislabeled_images
from assignment1_4_1 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)   # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

'''
# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
num_px_ = train_x_orig.shape[2]
num_channel = train_x_orig.shape[3]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px_) + ", " + str(num_channel) + ")")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
'''

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # the "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize data to have feature values between 0 and 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255

'''
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
'''

# constants defining the model
n_x =  12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# 两层神经网络
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a two_layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X - input data, of shape (n_x, number of examples)
    Y - true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims - dimensions of the layers (n_x, n_h, n_y)
    num_iterations - number of iterations of the optimization loop
    learning _rate - learning rate of the gradient descent update rule
    print_cost - If set t True, this will print the cost every 100 iterations

    Returns:
    parameters - a dictionary containing W1, W2, b1, b2
    """

    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        # print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

'''
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost = False)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
'''

layers_dims = [12288, 20, 7, 5, 1]

# L层神经网络
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments
    X -
    Y -
    layers_dims -
    learning_rate - 
    num_iterations -
    print_cost -

    Returns:
    parameters -
    """

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 training example
        if print_cost and i % 100 ==0:
            print("Cost after iterarions %i: %f" % (i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

#print_mislabeled_images(classes, test_x, test_y, pred_test)
#print("trained parameters is : " + str(parameters))

# test with your own image
def predict_img():
    my_image = "my_image"
    my_label_y = [1, 1, 1, 1]
    num_px = 64

    for i in range(1, 5):
        fname = "images/" + my_image + str(i) + ".jpg"
        image = np.array(ndimage.imread(fname, flatten = False))
        my_image = scipy.misc.imresize(image, size = (num_px, num_px)).reshape((num_px * num_px * 3, 1))
        my_predicted_image = predict(my_image, my_label_y[i-1], parameters)
        
        plt.imshow(image)
        plt.show()
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")

    return

predict_img()
