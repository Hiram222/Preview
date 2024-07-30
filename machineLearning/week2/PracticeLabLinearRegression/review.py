import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
x_train, y_train = load_data()
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()
