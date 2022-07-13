import matplotlib.pyplot as plt
import VGG
import tensorflow as tf
import cv2
mymodel = VGG.VGG16()

Fruit =r'D:\Fruit\text\3.jpeg'
X_train = []
img = cv2.imread(Fruit)
img = tf.image.resize(img, [128, 128])
X_train.append(img)
X_train=tf.convert_to_tensor(img,dtype=tf.float32)

mymodel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

mymodel.build(input_shape=(None,128,128,3))
mymodel.load_weights('sbie.h5')
# mymodel.summary()
X_train=X_train[tf.newaxis,]
print(X_train.shape)
a = mymodel.predict(X_train)
b = tf.argmax(a,axis=1)
c=tf.print(b)

if b == 0:
    print("苹果")
if b == 1:
    print("香蕉")
if b == 2:
    print("橘子")
if b == 3:
    print("西瓜")

plt.pause(1)
plt.close()

