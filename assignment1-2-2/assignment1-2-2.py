import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#print(train_set_x_orig)

# Example of a picture
index=208
'''plt.imshow(train_set_x_orig[index])
pylab.show()
print("y="+str(train_set_y[:,index])+", it's a '"+classes[np.squeeze(train_set_y[:, index])].decode('utf-8')+"' picture.")
'''
# Number of examples
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

'''print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))'''

# Reshape the training and test examples
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3]).T

'''print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))'''

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

# sigmoid
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

'''print('sigmoid([0,2])='+str(sigmoid(np.array([0,2]))))'''

# initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape(dim,1)for w and initializes b to 0

    Argument:
    dim - size of the w vector we want (or number of parameters in this case)

    Return:
    w - initialized vector of shape (dim,1)
    b - initialized scalar (corresponds to the bias)
    """
    w=np.zeros((dim,1))
    b=0

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b

'''dim=2
w,b=initialize_with_zeros(dim)
print('w='+str(w))
print('b='+str(b))'''

# propagate
def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w - weights, a numpy array of size (num_px*num_px*3,1)
    b - bias, a scalar
    X - data of size (num_px*num_px*3, number of examples)
    Y - true 'label' vector (containing 0 if non-cat, 1 if cat)of size (1, number of examples)

    Return:
    cost - negative log-likelihood cost for logistic regression
    dw - gradient of the loss with respect to w, thus same shape as w
    db - gradient of the loss with respect to b, thus same shape as b
    """

    m=X.shape[1]

    A=sigmoid(np.dot(w.T,X)+b)
    #cost=-1/m*(np.dot(Y,np.log(A.T)) + np.dot(np.ones((1,m))-Y,np.log(np.ones((m,1))-A.T)))
    cost=-1/m*(np.dot(Y,np.log(A.T)) + np.dot((1-Y),np.log(1-A.T)))

    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.dot(np.ones((1,m)),(A-Y).T)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    db=np.squeeze(db)
    cost=np.squeeze(cost)
    assert(cost.shape == ())

    grads={'dw':dw,
           'db':db}

    return grads,cost

'''w,b,X,Y=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
grads,cost=propagate(w,b,X,Y)
print('dw='+str(grads['dw']))
print('db='+str(grads['db']))
print('cost='+str(cost))'''

# optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w - weights, a numpy array of size (num_px*num_px*3, 1)
    b - bias, a scalar
    X - data of shape (num_px*num_px*3, number of examples)
    Y - true 'label' vector (containing 0 if non-cat, 1 if cat),of shape(1,nubmer of examples)
    num_iterations - number of iterations of the optimization loop
    learning_rate - learning rate of the gradient descent update rule
    print_cost - True to print the loss every 100 steps

    Returns:
    params - dictionary containing the weights w and bias b
    grads - dictionary containing the gradients of the weights and bias with respect to the cost function
    costs - list of all the costs computed during the optimization, this will be used to plot the learning curve
    """

    costs = []

    for i in range(num_iterations):

        grads,cost = propagate(w, b , X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 ==0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {'w':w,
              'b':b}

    grads = {'dw':dw,
             'db':db}

    return params, grads, costs

'''params, grads, costs = optimize(w, b, X, Y)
print ('w='+str(params['w']))
print ('b='+str(params['b']))
print ('dw='+str(grads['dw']))
print ('db='+str(grads['db']))'''

# predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias a scalar
    X - data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction - a numpy array (vector) containing predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):

        Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

'''print("predictions = " + str(predict(w, b, X)))'''

# model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the funciton you have implemented previously

    Arguments:
    X_train - training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train - training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test - test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test - test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate - hyperparameter representing the learning rate of used in the update rule of optimize()
    print_cost - set to true to print the cost every 100 iterations

    Return:
    d - dictionary containing information about the model
    """

    w, b = initialize_with_zeros(train_set_x_flatten.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print('train accuracy: {} %'.format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print('test accuracy: {} %'.format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {'costs':costs,
         'Y_prediction_test':Y_prediction_test,
         'Y_prediction_train':Y_prediction_train,
         'w':w,
         'b':b,
         'learning_rate':learning_rate,
         'num_iterations':num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = 0.005, print_cost = False)

# Example of a picture that was wrongly classified
'''index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
pylab.show()
print ("y=" + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") + "\"")
'''

# plotfunc
def plotfunc():
    # Plot learning curve (with costs)
    '''costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('learning rate = ' + str(d['learning_rate']))
    plt.show()'''


    learning_rates = [0.05, 0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')
    
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    
    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    return

# Predict real cat image
def predict_real():
    my_image = "my_image.jpg"

    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten = False))
    my_image = scipy.misc.imresize(image, size = (num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d['w'], d['b'], my_image)

    plt.imshow(image)
    pylab.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image))].decode("utf-8") + "\"")


#plotfunc()
predict_real()
