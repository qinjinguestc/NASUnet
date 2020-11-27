from loadMHA.readMHATrain import *
import numpy as np
from matplotlib import pyplot as plt

brats = BRATS2015()

count = np.zeros(155)

for i in range(brats.total_batch):#220
    count = np.zeros(155)
    for j in range(155):
        img, lable = brats.next_train_batch(1)
        if np.max(lable) != 0:
            # count[j]代表第i个mha图中有肿瘤的图片
            count[j] += 1
            print(i, j, count[j])

for i in range(155):
    print("count"+str(i)+':', count[i])

# 创建柱状图
# 第一个参数为柱的横坐标
# 第二个参数为柱的高度
# 参数align为柱的对齐方式，以第一个参数为参考标准
plt.bar(range(155), count.tolist(), align='center', yerr=0.000001)

# 设置柱的文字说明
# 第一个参数为文字说明的横坐标
# 第二个参数为文字说明的内容
# plt.xticks(range(155), ran)
plt.show()
