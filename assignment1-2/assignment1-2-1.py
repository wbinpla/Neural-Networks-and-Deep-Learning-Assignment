'''import math
def basic_sigmoid(x):
    s=1/(1+math.exp(-x))
    return s

print(basic_sigmoid([1,2,3]))'''


import numpy as np
def basic_sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

'''x=np.array([1,2,3])
print(basic_sigmoid(x))'''

def sigmoid_derivative(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return ds

'''x=np.array([1,2,3])
print("sigmoid_derivative(x)="+str(sigmoid_derivative(x)))'''

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

def image2vector(image):
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

'''print('image2vector(image)='+str(image2vector(image)))'''

x=np.array([[0,3,4],[1,6,4]])
def normalizeRows(x):
    x_norm=np.linalg.norm(x,ord=2,axis=0,keepdims=True)
    x=x/x_norm
    return x

'''print('normalizeRows(x)='+str(normalizeRows(x)))'''

x=np.array([[9,2,5,0,0],[7,5,0,0,0]])
def softmax(x):
    x_exp=np.exp(x)
    x_sum=np.sum(x_exp,axis=1,keepdims=True)
    s=x_exp/x_sum
    return s

'''print('softmax(x)='+str(softmax(x)))'''

'''import time
x1=np.array([[9,2,5,0,0,7,5,0,0,0,9,2,5,0,0]])
x2=np.array([[9,2,2,9,0,9,2,5,0,0,9,2,5,0,0]])

tic=time.process_time()
dot=np.dot(x1.T,x2)
toc=time.process_time()
print('dot='+str(dot)+'\n---computation time='+str(1000*(toc-tic))+'ms')'''

yhat=np.array([.9,.2,.1,.4,.9])
y=np.array([1,0,0,1,1])
def L1(yhat,y):
    loss=np.sum(np.abs(yhat-y))
    return loss

print("L1="+str(L1(yhat,y)))

def L2(yhat,y):
    loss=np.dot((yhat-y),(yhat-y))
    return loss

print("L2="+str(L2(yhat,y)))
