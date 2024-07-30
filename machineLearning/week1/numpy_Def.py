import numpy as np
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

array_1d = np.array([1.0, 2.0, 3.0])
print("一维数组的形状:", array_1d.shape)  # 输出: (3,)
#一维数组不存在什么行列，就只会统计数量
array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("二维数组的形状:", array_2d.shape)  # 输出: (2, 3)
#二维数组会按照行优先进行打印，一共有两行，三列