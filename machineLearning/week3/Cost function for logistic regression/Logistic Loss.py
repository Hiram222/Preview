import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

soup_bowl()
#数据集
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
#当损失函数是平方差的情况
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()
#损失函数为log的图像
plt_two_logistic_loss_curves()
#损失函数为log的损失函数
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)


