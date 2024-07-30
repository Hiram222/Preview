import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)

#实验中，将两组数据按照x,y轴进行拆分
#x_train表示的是x轴上的数据
#数据初始化
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"方法二：Number of training examples is: {m}")
#Numpy 数组有一个 .shape 参数。返回数组的维度信息
#其中每个维度都有一个条目。x_train.shape[0]是数组的长度和示例的数量，

# m is the number of training examples
m = len(x_train)
print(f"方法一： Number of training examples is: {m}")
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")



#必须要将w和b转换为整数
print("Try w=200 and b=100 ")
w = input("输入w：")
w=int(w)
b = input("输入b：")
b=int(b)
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)#初始化一个与输入数据 x 大小相同的数组
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb#依次放入预测数据


tmp_f_wb = compute_model_output(x_train, w, b,)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

#对应的预测值
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")