import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

#数据集(存在矩阵中，一行表示一个示例)
X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#打印数据集
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

#逐个元素的单个预测
def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p
print("采用逐个相乘再相加的方式对向量进行运算")
x_vec = X_train[0,:]#取数据集第一行
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")#该模型下，第一示例的预测结果

#使用内置库进行运算
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

print("采用dot函数对向量进行运算")
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#成本函数
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  #计算当前预测结果
        cost = cost + (f_wb_i - y[i]) ** 2  #平方差
    cost = cost / (2 * m)  # scalar
    return cost
# Compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

#偏导数计算
def compute_gradient(X, y, w, b):

    m, n = X.shape  # 数据的行、列
    dj_dw = np.zeros((n,))#初始化一个全是零的一维数组
    dj_db = 0.
    #i是第i个特性
    for i in range(m):#（1）
        err = (np.dot(X[i], w) + b) - y[i] #（2）误差
        for j in range(n):#（3）
            dj_dw[j] = dj_dw[j] + err * X[i, j]#（4）累加
            #dj_dw[j]是wj的偏导，进行累加
            dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

#Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

#梯度下降
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
        Args:
          X数据集
          y示例标签
          w_in 初始模型参数wi
          b_in 初始模型参数b
          cost_function       成本函数
          gradient_function   梯度函数
          alpha (float)       学习率
          num_iters (int)     迭代次数
        Returns:
            梯度下降后的参数
          """
    J_history = []
    w = copy.deepcopy(w_in)  #深拷贝，避免修改，导致错误
    b = b_in
    for i in range(num_iters):#迭代num_iters次
        dj_db, dj_dw = gradient_function(X, y, w, b)
        #更新参数
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None
        #保存代价
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:#每10%打印一次运行结果
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history  # return final w,b and J history for graphing

#初始化参数设置
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
#执行梯度下降
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    # plot cost versus iteration

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()