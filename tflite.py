#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:48:01 2021

@author: jimmy
"""

import tensorflow as tf
import cv2
from tensorflow.keras import layers,optimizers,losses

batchsz = 32

def preprocess(x,y):
    

    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=10)
    y = tf.expand_dims(y, axis = 0)
 
    return x,y
    
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).batch(batchsz).shuffle(1000)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(batchsz)

net = tf.keras.applications.DenseNet121(weights = 'imagenet',include_top = False)
# net.summary()

for layer in net.layers :
    layer.trainable = False
    
net1 = tf.keras.Sequential()
net1.add(net)
net1.add(layers.Dense(1024,activation = 'relu'))
net1.add(layers.BatchNormalization())
net1.add(layers.Dropout(rate=0.2))
net1.add(layers.Dense(10))

#net1.summary()
net1.build(input_shape = (4,32,32,3))

net1.compile(optimizer = optimizers.Adam(lr=1e-4),
    loss = losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy'])
    
net1.fit(train_db,validation_data = test_db ,epochs = 1)

tf.saved_model.save(net1,"/home/jimmy/Desktop/save_model")
   