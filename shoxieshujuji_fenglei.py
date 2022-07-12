import numpy as np
import os
import cv2
dir = 'D:\Fruit'
X_train = []
Y_train = []
Fruit = r'D:\Fruit'
for root, dirs, files in os.walk(Fruit):
    for file in files:
        path_file = os.path.join(root, file)  # os.path.join(a,b)函数是指把'a',与'b'的路径拼接起来
        if path_file.endswith(".jpg"):  # endswith() 方法用于判断字符串是否以指定后缀结尾
            img = cv2.imread(path_file)
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

