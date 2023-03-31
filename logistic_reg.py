import numpy as np

def Sigmoid(theta, X):
    return 1/(1+np.exp(-X@theta))

def Cost(theta, X, y):
    h = Sigmoid(theta, X)
    N = len(y)
    return (1/N)*np.sum((-y)*np.log(h)-(1-y)*(np.log(1-h)))

def Gradient(X, Y, theta):
    h = Sigmoid(theta, X)
    N = len(Y)
    return (1/N)*X.T@np.sum(h-Y)

def Descent(X, y, theta_0, learning_rate=0.001, threshold = 0.05):
    theta = theta_0
    costs = []
    thetas = []
    
    while np.linalg.norm(Gradient(X,y, theta))>threshold:
        theta -= learning_rate*Cost(theta, X,y)
        costs.append[Cost(theta, X, y)]
        thetas.append(theta.copy())
    return theta, costs, thetas