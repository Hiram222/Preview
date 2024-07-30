i=1
while i<5: #条件判断后，用:结束
    print(i)
    i+=1
else:
    print("--------------")


for letter in 'Python':  # 遍历字符串
    print("当前字母: %s" % letter)

fruits = ['banana', 'apple', 'mango']
for fruit in fruits:  #遍历列表
    print('当前水果: %s' % fruit)

for k in range(2,3):
    print(k)

print("--------------")

fruits = ['banana', 'apple', 'mango']
for index in range(len(fruits)):
    print('当前水果 : %s' % fruits[index])

# !/usr/bin/python
# -*- coding: UTF-8 -*-
print("--------------")

# 输出 Python 的每个字母
for letter in 'Python':
    if letter == 'h':
        pass
    print('当前字母 :', letter)



def printinfo(name, age):
    "打印任何传入的字符串"
    print("Name: ", name)
    print("Age ", age)
    return name
# 调用printinfo函数
printinfo(age=50, name="miki")
print(printinfo(1,1))
