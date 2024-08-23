import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging

# 设置日志级别为错误，以减少输出信息
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')  # 除错误外不显示日志
tf.autograph.set_verbosity(0)

# 加载数据
X, Y = load_coffee_data()
print(X.shape, Y.shape)

plt_roast(X, Y)
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

# 数据标准化
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# 扩展训练数据集
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)

# 模型构建
tf.random.set_seed(1234)
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation='sigmoid', name='layer1'),
    Dense(1, activation='sigmoid', name='layer2')
])

model.summary()

# 输出每层的参数数量
L1_num_params = 2 * 3 + 3
L2_num_params = 3 * 1 + 1
print("L1 params =", L1_num_params, ", L2 params =", L2_num_params)

# 打印权重和偏置
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# 编译模型
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

# 训练模型
model.fit(Xt, Yt, epochs=10)

# 更新权重
W1 = np.array([
    [-8.94, 0.29, 12.89],
    [-0.17, -7.34, 10.79]
])
b1 = np.array([-9.87, -9.28, 1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]
])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1, b1])
model.get_layer("layer2").set_weights([W2, b2])

# 预测
X_test = np.array([
    [200, 13.9],  # positive example
    [200, 17]     # negative example
])
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

# 二元决策
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

# 可视化
plt_layer(X, Y.reshape(-1,), W1, b1, norm_l)
plt_output_unit(W2, b2)
netf = lambda x: model.predict(norm_l(x))
plt_network(X, Y, netf)
