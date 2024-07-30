import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')
#数据集
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)
#展示
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()



def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape #获得实例数和维数 m=6,n=2
    dj_dw = np.zeros((n,))  # 初始化
    dj_db = 0.
    for i in range(m): #有m个实例
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  #求当前的预测值
        err_i = f_wb_i - y[i]  #差
        for j in range(n): #n个参数
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  #每个参数实现累加
        dj_db = dj_db + err_i#实现累加
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw
#检查梯度函数的实现。
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

#逻辑回归梯度下降代码实现
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
      X (ndarray (m,n)   :数据集\ y (ndarray (m,))   :目标值
      w_in (ndarray (n,)):初始w\ b_in (scalar)      : 初始b
      alpha (float)      :学习率\ num_iters (scalar) :迭代次数
    """
    #用于存储每一次迭代的w和j
    J_history = []
    w = copy.deepcopy(w_in)  # 深拷贝
    b = b_in
#执行迭代
    for i in range(num_iters):
        #计算偏导
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        #更新
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        #记录迭代过程
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost_logistic(X, y, w, b))
        #每10%打印运行成果
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing

#输入初始化数据
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000
#查看结果
w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability
plt_prob(ax, w_out, b_out)
# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)
# Plot the decision boundary
x0 = -b_out/w_out[1]
x1 = -b_out/w_out[0]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()


#使用单变量数据集
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )


