import numpy as np

import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)
    plt.pause(0.001)
    ln[0].remove()


def sigmoid(score):
    return 1/(1 + np.exp(-score))


def calculate_error(line_parameters, points, labels):
    m = points.shape[0]
    p = sigmoid(np.dot(points, line_parameters))
    cross_entropy = (-1/m)*(np.dot(np.log(p).T, labels) + np.dot(np.log(1-p).T, (1-labels)))
    return cross_entropy


def gradient_descent(line_parameters, points, labels, learning_rate):
    m = points.shape[0]
    #print(line_parameters)
    for i in range(20000):
        p = sigmoid(np.dot(points, line_parameters))
        gradient = (learning_rate/m)*np.dot(points.T, (p - labels))
        #print(gradient)
        line_parameters -= gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b/w2 + (x1*(-w1/w2))
        if x2[0] < 800 and x2[1] < 800:
            draw(x1, x2)
        else:
            print(i, x2)



np.random.seed(0)

n_pts = 100
bias = np.ones(n_pts)

random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(50, 10, n_pts)
#random_x1_values = np.random.normal(10, 2, n_pts)
#random_x2_values = np.random.normal(12, 2, n_pts)
top_region = np.array([random_x1_values, random_x2_values, bias]).T
#bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(30, 10, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

# choose a random starting line
#w1 = -0.2
#w2 = -0.35
#b = 3.5

line_parameters = np.array([np.zeros(3)]).reshape(3, 1)

#x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
#x2 = -b/w2 - x1*(w1/w2)

#linear_combination = np.dot(all_points, line_parameters)
#print(linear_combination)

#probabilities = sigmoid(linear_combination)
#print(probabilities)

labels = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)
#print(calculate_error(line_parameters, all_points, labels))




_, ax = plt.subplots(figsize=(4, 4))
plt.xlim(0, 15)
plt.ylim(0, 80)
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, labels, 0.002)
plt.show()


