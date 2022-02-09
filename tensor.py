import  tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import  numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import os
print(tf.__version__)
image_dates = glob('./img/*/*.jpg')
class_name = ['iu', 'suzy', 'uin']
dic = {"iu": 0, 'suzy': 1, 'uin': 2}

X = [] # 이미지 저장 리스트
Y = [] #라벨 저장 리스트
for imagename in image_dates:
    image = Image.open(imagename) # 이미지 개별로 불러오기
    image = image.resize((128, 128)) #이미지 사이즈 조절 (128, 128)
    image = np.array(image) #이미지 수열로 변화
    X.append(image) # X리스트에 이미지 넣기
    label = imagename.split('\\')[1]
    Y.append(dic[label]) # 사진별 라벨 Y 리스트에 넣기


X = np.array(X)
Y = np.array(Y)

train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, shuffle= True, random_state= 44)

train_labels = train_labels[..., tf.newaxis]
test_labels = test_labels[..., tf.newaxis]

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

#training set의 각 class 별 image 수 확인
unique, count = np.unique(np.reshape(train_labels,(72,)), axis= -1, return_counts=True)
print(dict(zip(unique, count)))

#test set 의 각 class 별 image 수 확인
unique, count = np.unique(np.reshape(test_labels,(18,)), axis= -1, return_counts=True)
print(dict(zip(unique, count)))

N_TRAIN = train_images.shape[0]
N_TEST = test_images.shape[0]

print(N_TRAIN, N_TEST)

# pixel 값을 0~1 사이 범위로 조정
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.

# label 을 one_hot encoding
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
print(train_images.ndim)
print(train_labels.ndim)

learning_rate = 0.01
N_EPORCHS =10
N_BATCH = 1
N_CLASS = 5
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(72).batch(N_BATCH).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(18).batch(N_BATCH)

def creadte_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size= 3, activation='relu',
                                  padding='SAME',
                                  input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D(padding="SAME"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                  padding='SAME',
                                  input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D(padding="SAME"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',
                                  padding='SAME',
                                  input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D(padding="SAME"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(3, activation='softmax'))
    return model

model = creadte_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

steps_per_epoch = N_TRAIN// N_BATCH
validation_steps = N_TEST// N_BATCH
print(steps_per_epoch, validation_steps)

history = model.fit(train_dataset, epochs=N_EPORCHS, steps_per_epoch= steps_per_epoch,
                    validation_data= test_dataset, validation_steps= validation_steps)
model.evaluate(test_dataset)

