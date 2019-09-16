#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:38:49 2019

@author: david
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import healpy as hp
import os
from pathlib import Path
parent_path=os.fspath(Path(__file__).parent.parent)

data_path=parent_path+'/7mindata/npymats'

#Image reading

size=150
nmaps=2

data_full=[]
data_label=[]
fdata_full=[]
fdata_label=[]

num_cleanpatches_detection=int(0.75*3072) #85% brightest patches in the sky
num_forepatches_detection=3072-num_cleanpatches_detection

numct=num_cleanpatches_detection+150
numft=num_forepatches_detection+150

foreorder=np.load('fore_order.npy') #weak foregrounds go FIRST

cleanindices=np.array(foreorder[:numct+100])
foreindices=np.array(foreorder[(3072-numft):])
        
#Weak foreground loading

for i in range(1,nmaps+1):
    np.random.shuffle(cleanindices)
    for rem in cleanindices:
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        data_full.append(np.array(full))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        data_label.append(np.array(label))
        
data_full=np.array(data_full).reshape(len(cleanindices)*nmaps,size,size,1)
data_label=np.array(data_label).reshape(len(cleanindices)*nmaps,size,size,1)

#Strong foreground loading
        
for i in range(1,nmaps+1):
    np.random.shuffle(foreindices)
    for rem in foreindices:
        full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
        fdata_full.append(np.array(full))
        label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
        fdata_label.append(np.array(label))

fdata_full=np.array(fdata_full).reshape(len(foreindices)*nmaps,size,size,1)
fdata_label=np.array(fdata_label).reshape(len(foreindices)*nmaps,size,size,1)

#We dont need validation statistics in the strong fore. We dont want to miss images.
lenj=(numct-25)*nmaps

print('Data loaded')

#################################################################
#                MODEL WEAK FOREGROUNDS
#################################################################

##CNN model
model=tf.keras.Sequential()
model.add(layers.Conv2D(16, (7, 7), input_shape=(size,size,1), strides=(1,1), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (7, 7), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))  
model.add(layers.Activation('relu'))
model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(1, (1, 1), activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.summary()

history=model.fit(data_full[:lenj,:,:,:], data_label[:lenj,:,:,:], epochs=12, batch_size=50,
          validation_data=(data_full[lenj:,:,:,:], data_label[lenj:,:,:,:]))

model.save('model75_weakforegrounds_14ep')

history_dict=history.history
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################################################################
#                MODEL STRONG FOREGROUNDS
#################################################################

#CNN model
modelr=tf.keras.Sequential()
modelr.add(layers.Conv2D(16, (7, 7), input_shape=(size,size,1), strides=(1,1), padding='same'))
modelr.add(layers.BatchNormalization())
modelr.add(layers.Activation('relu'))
modelr.add(layers.Conv2D(32, (7, 7), padding='same'))
modelr.add(layers.BatchNormalization())
modelr.add(layers.Activation('relu'))
modelr.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))  
modelr.add(layers.Activation('relu'))
modelr.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
modelr.add(layers.BatchNormalization())
modelr.add(layers.Activation('relu'))
modelr.add(layers.Conv2D(1, (1, 1), activation='sigmoid'))

modelr.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
modelr.summary()

historyr=modelr.fit(fdata_full, fdata_label, epochs=24, batch_size=50)

modelr.save('model75_strongforegrounds_24ep')

history_dict=historyr.history
acc = history_dict['binary_accuracy']
loss = history_dict['loss']

epochs = range(1, len(acc) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()