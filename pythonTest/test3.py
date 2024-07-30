#
#
# # 打开一个文件
# f = open("./foo.txt", "w")
# num=f.write("Python 是一个非常好的语言。\n是的，的确非常好!!\n" )
# print(num)
# # 关闭打开的文件
# f.close()
#
#
# # 打开一个文件
# f = open("./foo.txt", "r")
# str = f.read()
# print(str)
# # 关闭打开的文件
# f.close()


# 打开一个文件
f = open("../foo.txt", "r")

str = f.readline()
print(str)

# 关闭打开的文件
f.close()