from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers
(train_img,train_label),(test_img,test_label)=datasets.mnist.load_data()
#since CNN only takes 4D tensor as input we should reshape the input
train_img=train_img.reshape((60000,28,28,1))
test_img=test_img.reshape((10000,28,28,1))
#normalization so that the value lies between 0 and 1
train_img=train_img/255.0
test_img=test_img/255.0
#equation model for convolutional base
model=models.Sequential()
#create hidden layers
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.summary() #function provides valueable information about the model like parameters, size etc.
# we feed the o/p of size (sample, 3,3, 64) to the dense layer for classification
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

#specify optimiser, loss function and metric to track
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(train_img,train_label,epochs=15) #default batch size=32
#evaluate the accuracy
test_loss,test_acc= model.evaluate(test_img,test_label)
print(test_acc)
# to test the code and make predictions import numpy as np and..
#make predictions
#p=model.predict(img_test)
#np.argmax(p[0])
######

