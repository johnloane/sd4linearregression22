import numpy as np

import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)


def sigmoid(score):
    return 1/(1 + np.exp(-score))


np.random.seed(0)

n_pts = 100
bias = np.ones(n_pts)

random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(50, 10, n_pts)

top_region = np.array([random_x1_values, random_x2_values, bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(30, 10, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

# choose a random starting line
w1 = -0.2
w2 = -0.35
b = 3.5

line_parameters = np.array([w1, w2, b]).reshape(3, 1)

x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b/w2 - x1*(w1/w2)

linear_combination = np.dot(all_points, line_parameters)
#print(linear_combination)

probabilities = sigmoid(linear_combination)
#print(probabilities)




_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()


