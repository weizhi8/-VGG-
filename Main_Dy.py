import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import VGG
# 使用GPU加速的，没有GPU的记得把下面注释掉
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

x_train = np.load('x_train.npy',allow_pickle=True)
y_train = np.load('y_train.npy',allow_pickle=True)
x_test = np.load('x_val.npy',allow_pickle=True)
y_test = np.load('y_val.npy',allow_pickle=True)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = VGG.VGG16()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
mymodel=model.fit(x_train,y_train,batch_size=20,epochs=100,validation_data=(x_test,y_test))
model.summary()
model.save_weights("sbie.h5")

plt.figure(figsize=(16, 9))
plt.suptitle('loss and accuracy', fontsize=14, fontweight="bold")
#训练集损失率
plt.subplot(2,2,1)
plt.title(" train Loss ")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(mymodel.history['loss'])
#测试集损失率

plt.subplot(2,2,2)
plt.title(" text Loss ")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(mymodel.history['val_loss'])
#测试集准确率

plt.subplot(2,2,3)
plt.title("train accuracy")
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(mymodel.history['sparse_categorical_accuracy'])

#测试集准确率

plt.subplot(2,2,4)
plt.title("text accuracy")
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(mymodel.history['val_sparse_categorical_accuracy'])
plt.show()

# #输出测试测试集数据
# result=model.evaluate(x_test,y_test)
# print('='*50)
# print(result)
# #输出训练集测试数据
# result=model.evaluate(x_train,y_train)
# print('='*50)
# print(result)


