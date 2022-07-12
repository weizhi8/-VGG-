import sklearn
import  tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

a=np.load('x.npy',allow_pickle=True)
b=np.load('y.npy',allow_pickle=True)
data=a
label=b
# tf.cast(data,tf.float32)
seed = 785
#生成随机种子
np.random.seed(seed)
x_train,x_val,y_train,y_val=sklearn.model_selection.train_test_split(data,label,test_size=0.20,random_state=0)


# 因为本人电脑带不动224*224故改成128*128，最好是224*224比较贴合VGG神经网络
x_train = tf.image.resize(x_train, [128, 128])
x_val = tf.image.resize(x_val, [128, 128])
# x_train = x_train  / 255
# x_val = x_val / 255
# print(x_test)

# print(y_train)
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)
np.save('x_test.npy',x_val)
np.save('y_test.npy',y_val)

