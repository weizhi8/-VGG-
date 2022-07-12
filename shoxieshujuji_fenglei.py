import random
import numpy as np

import matplotlib.pyplot as pot
import tensorflow as tf
from tensorflow import keras
import glob
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import os
import cv2
dir = 'D:\Fruit'
X_train = []
Y_train = []
Fruit = r'D:\Fruit'
                                                                                                                # os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
                                                                                                                #     top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
                                                                                                                #         root 所指的是当前正在遍历的这个文件夹的本身的地址
                                                                                                                #         dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
                                                                                                                #         files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
                                                                                                                #     topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
                                                                                                                #     onerror -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。
                                                                                                                #     followlinks -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录。
i = 1
for root, dirs, files in os.walk(Fruit):
    for file in files:
        path_file = os.path.join(root, file)  # os.path.join(a,b)函数是指把'a',与'b'的路径拼接起来
        # print(path_file)
        # 上面这个写法是遍历文件，遍历文件夹方法如下：
        # for d in dirs:
        #     print(os.path.join(root, d))
        if path_file.endswith(".jpg"):  # endswith() 方法用于判断字符串是否以指定后缀结尾
            img = cv2.imread(path_file)
            # img = tf.image.resize(img, [128, 128])
            img = np.array(img)
            img = img.astype(np.uint8)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            X_train.append(img)
            kiy = os.path.basename(root)  # os.path.basename(root)是返回root的的文件夹名字
            if kiy == "apple":
                Y_train.append("0")
            if kiy == "banana":
                Y_train.append("1")
            if kiy == "tangerine":
                Y_train.append("2")
            if kiy == "watermelon":
                Y_train.append("3")
np.save("y.npy",Y_train)
np.save("x.npy",X_train)







# 将np数据转为张量
# a = np.load("x_train.npy")
# b=tf.convert_to_tensor(a,dtype=tf.float32)
# print(b)
# a=np.expand_dims(Y_train,axis=1)
# a1=np.expand_dims(a,axis=1)
# a2=np.expand_dims(a1,axis=1)
# Y_train=a2
# print(Y_train.ndim)
# print(Y_train)
# print('apple={}'.format(apple))
# print('banana={}'.format(banana))
# print('tangerine={}'.format(tangerine))
# # print('watermelon={}'.format(watermelon))
# files = os.listdir(dir)
# files.sort()  # .sort()函数如上，其实是一个排序函数
# # print(files)
# a = os.walk(dir)
# print(a)
# print(dir)
# data = pd.read_csv(D:\Fruit)
# model = tf.keras.Sequential()
# model.add(layers.Dense(32,activation='relu'))
# model.add(layers.Dense(32,activation='relu'))
# model.add(layers.Dense(10,activation='softmax')) # 10的意思是我们要给出这个数据是“1-10”每个数子的概率
#
# model.compile(optimizer = tf.keras.optimizers.Adm(0.005),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_valid,y_valid))
