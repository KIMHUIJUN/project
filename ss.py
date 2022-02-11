import  tensorflow as tf
from sklearn.model_selection import train_test_split
import  numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
print(tf.__version__)
image_datas = glob('./img/*/*.jpg')
class_name = ['iu', 'suzy', 'uin']
dic = {"iu": 0, 'suzy': 1, 'uin': 2}

X = [] # 이미지 저장 리스트
Y = [] #라벨 저장 리스트

for imagename in image_datas:
    image = Image.open(imagename) # 이미지 개별로 불러오기
    image = image.resize((128, 128)) #이미지 사이즈 조절 (128, 128)
    image = np.array(image) #이미지 수열로 변화
    X.append(image) # X리스트에 이미지 넣기
    label = imagename.split('\\')[1]
    Y.append(dic[label]) # 사진별 라벨 Y 리스트에 넣기

print(len(X))
X = np.array(X)
Y = np.array(Y)

train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, shuffle= True, random_state= 44)

train_labels = train_labels[..., tf.newaxis]
test_labels = test_labels[..., tf.newaxis]

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# pixel 값을 0~1 사이 범위로 조정
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

plt.figure(figsize= (10, 10))
for i in range(9):
    plt.subplot(3, 3 , i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    a = train_labels[i]
    plt.xlabel(class_name[a[0]])
plt.show()

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(128, 128,3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ]
)

model.compile(optimizer= 'adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
              metrics=['accuracy'])

#MODEl_DIR = './model'
#if not os.path.exists(MODEl_DIR):
 #   os.mldir(MODEl_DIR)

#modelpath = './model'
#checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath= modelpath, monitor='val_loss', verbose= 1, save_best_only= True)

model.fit(train_images, train_labels, epochs= 50, batch_size= 30)
model.save('my_model.h5')
print('\n Accuracy : %.4f' % (model.evaluate(test_images, test_labels)[-1]))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose= 2)
print('\n Test accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

def plot_image(i , predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    print(predictions_array)
    name_idx = true_label[0]
    print(name_idx)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
                                         100 *np.max(predictions_array),
                                         class_name[name_idx]),
               color = color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(3))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label[0]].set_color('blue')

i = 1
plt.figure(figsize= (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
