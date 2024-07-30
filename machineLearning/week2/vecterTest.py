import numpy as np    # it is an unofficial standard to use np for numpy
import time

#np.zeros 函数
# 功能：分配一个数组，并用零填充它。
# 语法：np.zeros(shape, dtype=float, order='C')
# shape：数组的形状，可以是一个整数或一个整数元组。
#zeros(4)和zeros((4,))都是创造一个维度为4的一维数组

# dtype（可选）：数组中元素的类型，默认为 float（即 float64）。
# order（可选）：存储元素的顺序，默认为按行（'C' 顺序）。

a = np.zeros(4);
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4);#使用随机值进行填充
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#一些数据创建例程不采用形状元组：
#np.arange 函数
# 功能：返回一个数组，数组中的值是一个范围内的等差数列。
# 语法：np.arange([start,] stop[, step,], dtype=None)
# start（可选）：序列的起始值，默认是 0。
# stop：序列的结束值（不包括该值）。
# step（可选）：两数之间的间隔，默认是 1。
# dtype（可选）：数组元素的类型，如果没有提供，将从其他参数推断。

#np.random.rand 函数
# 功能：返回一个给定形状的数组，数组中的值是均匀分布在 [0, 1) 区间的随机数。
# 语法：np.random.rand(d0, d1, ..., dn) d0, d1, ..., dn：定义数组的形状的整数。
a = np.arange(4.);
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#按照自身要求创造数组
a = np.array([5,4,3,2]);
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]);
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#a.shape返回数组维数，a.dtype返回数组中的数据类型

print()
a = np.arange(10)#按顺序创造
print(a)
#当你访问一个数组的单个元素时，返回的是一个标量，标量是没有shape的
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
#-1表明从后往前第一个
print(f"a[-1] = {a[-1]}")

#索引是从0~n-1的，超过索引范围就会报错
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

print()
#切片  values(start:stop:step)
a = np.arange(10)
print(f"a         = {a}")
#elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)
c = a[2:7:2];     print("a[2:7:2] = ", c)
c = a[3:];        print("a[3:]    = ", c)
c = a[:3];        print("a[:3]    = ", c)
c = a[:];         print("a[:]     = ", c)
#没有的元素直接省略（：都可以省略）


#数组上的运算
print()
a = np.array([1,2,3,4])
print(f"a             : {a}")
b = -a #取反
print(f"b = -a        : {b}")
b = np.sum(a)#数组求和
print(f"b = np.sum(a) : {b}")
b = np.mean(a)#求平均值
print(f"b = np.mean(a): {b}")
b = a**2 #求平方
print(f"b = a**2      : {b}")

#数组之间的运算
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

#数组之间类型不同，不可以相互运算
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])
#数组乘法
b = 5 * a
print(f"b = 5 * a : {b}")

#点积，返回的是一个值
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")


def my_dot(a, b):#通过for循环对数组的每一项相乘再相加
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

print()
#通过非常大范围的数组来证明numpy内置函数的效率
np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)
tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time
print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")
tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time
print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")
del(a);del(b)  #释放空间

# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)
print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
