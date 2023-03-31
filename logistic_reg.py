import numpy as np

def Sigmoid(theta, X):

    z = X@theta
    return 1/(1+np.exp(-z))

def Cost(theta, X, y):
    
    m = len(y)  
    h = Sigmoid(theta, X)  
    J = -1.0/m * np.sum(y*np.log(h) + (1-y)*np.log(1-h))  # compute cost
    return J


def Gradient(X, Y, theta):
    h = Sigmoid(theta, X)
    N = len(Y)
    return (1/N)*X.T@(h-Y)

def Descent(X, y, theta_0, lr=0.03, th = 0.03):
    theta = theta_0
    costs = []
    thetas = []
    

    
    while np.linalg.norm(Gradient(X,y, theta))>th:
        theta -= lr*Gradient(X, y, theta)
        costs.append(Cost(theta, X, y))
        thetas.append(theta.copy())
    return theta, costs, thetas