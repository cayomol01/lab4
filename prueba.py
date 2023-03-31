import numpy as np
import matplotlib.pyplot as plt
from logistic_reg import *



clean_data = np.genfromtxt('framingham.csv', delimiter=',',usecols=(0,1,3,4,5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15), skip_header=True)
mean = np.nanmean(clean_data, axis=0)	# Calculate mean of each column
nan_indices = np.where(np.isnan(clean_data))
clean_data[nan_indices] = 0





data = clean_data.copy()


for i in range(data.shape[1] - 1):
    data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
train_data = data[:int(0.6*len(data))]
test_data = data[int(0.6*len(data)):]



X_train = train_data[:, :-1]
y_train = train_data[:, -1].reshape(train_data.shape[0], 1)
X_test  = test_data[:, :-1]
y_test = test_data[:, -1].reshape(test_data.shape[0], 1)

X = data[:, :-1]
y = data[:, -1].reshape(data.shape[0], 1)


theta = np.zeros((X_train.shape[1], 1))
theta, costs, thetas = Descent(X_train, y_train, theta)
# Predict the test data





y_pred = Sigmoid(theta, X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred

# Accuracy and precision


print("Accuracy: ", np.sum(y_pred == y_test) / len(y_test))
print("Precision: ", np.sum(y_pred[y_pred == 1] == y_test[y_pred == 1]) / np.sum(y_pred == 1))



new_data = clean_data.copy()
# Add polynomial feature: Sys BP
new_data = new_data**2
# Normalize the data except
for i in range(new_data.shape[1] - 1):
    new_data[:, i] = (new_data[:, i] - np.mean(new_data[:, i])) / np.std(new_data[:, i])


new_data = np.hstack((np.ones((new_data.shape[0], 1)), new_data))
train_data = new_data[:int(0.4*len(new_data))]
test_data = new_data[int(0.4*len(new_data)):]

X = train_data[:, :-1]
y = train_data[:, -1].reshape(train_data.shape[0], 1)


theta = np.ones((X.shape[1], 1))
theta, costs, thetas = Descent(X, y, theta)


# Predict the test data
X_test = test_data[:, :-1]
y_test = test_data[:, -1].reshape(test_data.shape[0], 1)

y_pred = Sigmoid(theta, X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
# Accuracy and precision
print("\nAccuracy: ", np.sum(y_pred == y_test) / len(y_test))
print("Precision: ", np.sum(y_pred[y_pred == 1] == y_test[y_pred == 1]) / np.sum(y_pred == 1))

degrees = [1,2, 3, 4, 5]


t_l = []
cv_l = []


train_accuracies = []
train_precisions = []
cv_accuracies = []
cv_precisions = []

    
for degree in degrees:
    a = clean_data.copy()
    
    a = a**degree

    for i in range(a.shape[1] - 1):

        a[:, i] = (a[:, i] - np.mean(a[:, i])) / np.std(a[:, i])
        
    a = np.hstack((np.ones((a.shape[0], 1)), a))

        

    
    train_data = a[:int(0.5*len(a))]
    cv_data = a[int(0.5*len(a)):int(0.7*len(a))]
    test_data = a[int(0.7*len(a)):]

    X = train_data[:, :-1]
    y = train_data[:, -1].reshape(train_data.shape[0], 1)


    theta_0 = np.ones((X.shape[1], 1))
    
    theta, costs, thetas = Descent(X, y, theta_0)
    t_l.append(costs[-1])


    # Predict the test data
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(test_data.shape[0], 1)

    y_pred = Sigmoid(theta, X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    # Accuracy and precision
    train_accuracies.append(np.sum(y_pred == y_test) / len(y_test))
    train_precisions.append(np.sum(y_pred[y_pred == 1] == y_test[y_pred == 1]) / np.sum(y_pred == 1)) 
    
    cv_x = cv_data[:, :-1]
    cv_y = cv_data[:, -1].reshape(cv_data.shape[0], 1)
    
    theta, costs, thetas = Descent(cv_x, cv_y, theta_0)
    cv_l.append(costs[-1])
    
    y_pred = Sigmoid(theta, X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    
    cv_accuracies.append(np.sum(y_pred == y_test) / len(y_test))
    cv_precisions.append(np.sum(y_pred[y_pred == 1] == y_test[y_pred == 1]) / np.sum(y_pred == 1)) 
    
    
plt.plot(degrees, t_l, label='train')
plt.plot(degrees, cv_l, label='validation')
plt.legend()
plt.xlabel('polynomial degree')
plt.ylabel('accuracy')
plt.show()
    
best_degree = degrees[np.argmax(train_accuracies)]

# Add polynomial feature: Sys BP
new_data = new_data**best_degree
# Normalize the data except
for i in range(new_data.shape[1] - 1):
    new_data[:, i] = (new_data[:, i] - np.mean(new_data[:, i])) / np.std(new_data[:, i])


new_data = np.hstack((np.ones((new_data.shape[0], 1)), new_data))
train_data = new_data[:int(0.6*len(new_data))]
test_data = new_data[int(0.6*len(new_data)):]

X = train_data[:, :-1]
y = train_data[:, -1].reshape(train_data.shape[0], 1)


theta = np.ones((X.shape[1], 1))
theta, costs, thetas = Descent(X, y, theta)


# Predict the test data
X_test = test_data[:, :-1]
y_test = test_data[:, -1].reshape(test_data.shape[0], 1)

y_pred = Sigmoid(theta, X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
# Accuracy and precision

print(f"\nBest degree: {best_degree}")
print("\nAccuracy: ", np.sum(y_pred == y_test) / len(y_test))
print("Precision: ", np.sum(y_pred[y_pred == 1] == y_test[y_pred == 1]) / np.sum(y_pred == 1))

