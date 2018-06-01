import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

data = np.array([[np.array([1, 3, 4, 9]), 20],[np.array([1, 7, 2, 3]), 14],[np.array([1, 2, 2, 2]), 8],
                [np.array([1, 1, 4, 2]), 11],[np.array([1, 2, 4, 6]), 16],[np.array([1, 1, 1, 0]), 3]])

def train(epoch=10000,eta=0.001):
    weight = np.random.randint(50,size=len(data[0][0]))
    weight = 0.451 * weight
    print(weight)
    for i in range(epoch):
        #run on all data sample
        for j in range(len(data)):
            v = np.dot(data[j][0],weight)
            actual = v;
            desired = data[j][1]
            error= desired - actual
            rate = eta * error
            weight += rate * data[j][0]
    print(weight)
    return weight

train()