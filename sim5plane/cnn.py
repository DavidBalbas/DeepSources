# -*- coding: utf-8 -*-

"""
This script loads the training images, compiles a CNN architecture and trains it.
It is intended for images both with and without foregrounds.

Version: August 18, 2019
@author David Balbas
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


#Image reading

size=150
nmaps=2 #number of full maps to train the net. 2 is recommended (roughly 6150 images)
data_full=[]
data_full_fore=[]
data_label=[]
val_full=[]
val_label=[]
val_mfiltered=[]
val_pssmth=[]
val_full_fore=[]
val_mfiltered_fore=[]
lenj=2870*nmaps #length of the training set. 3072*nmaps-lenj=length of the validation set.
indices=np.arange(3071*nmaps)
np.random.shuffle(indices)
dataindices=indices[:lenj]
valindices=indices[lenj:]

#TODO: CHANGE THE PATHS

for j in dataindices:
    i=j//3071
    rem=j%3071
    #full=np.load('./npymats_fore/full_'+str(i)+'_'+str(rem)+'.npy')
    #data_full.append(np.array(full))
    full_fore=np.load('./npymats_fore/full_fore_'+str(i)+'_'+str(rem)+'.npy')
    data_full_fore.append(np.array(full_fore))
    label=np.load('./npymats_fore/segps_'+str(i)+'_'+str(rem)+'.npy')
    data_label.append(np.array(label))

        
for j in valindices:
    i=1+j//3071
    rem=j%3071
    full=np.load('./npymats_fore/full_'+str(i)+'_'+str(rem)+'.npy')
    val_full.append(np.array(full))
    mfiltered=np.load('./npymats_fore/mfiltered_'+str(i)+'_'+str(rem)+'.npy')
    val_mfiltered.append(np.array(mfiltered))
    pssmth=np.load('./npymats_fore/ps_'+str(i)+'_'+str(rem)+'.npy')
    val_pssmth.append(np.array(pssmth))
    label=np.load('./npymats_fore/segps_'+str(i)+'_'+str(rem)+'.npy')
    val_label.append(np.array(label))
    full_fore=np.load('./npymats_fore/full_fore_'+str(i)+'_'+str(rem)+'.npy')
    val_full_fore.append(np.array(full_fore))
    mfiltered_fore=np.load('./npymats_fore/mfiltered_fore_'+str(i)+'_'+str(rem)+'.npy')
    val_mfiltered_fore.append(np.array(mfiltered_fore))


#Up to this point: matrices loaded in data_(full,label,mfiltered), val_().
#Reshaping of lists to convert them into input tensors
    
data_full_fore=np.array(data_full_fore).reshape(lenj,size,size,1)
data_label=np.array(data_label).reshape(lenj,size,size,1)
val_full=np.array(val_full).reshape(3071*nmaps-lenj,size,size,1)
val_label=np.array(val_label).reshape(3071*nmaps-lenj,size,size,1)
val_full_fore=np.array(val_full_fore).reshape(3071*nmaps-lenj,size,size,1)

#COMPILATION AND TRAINING OF THE CNN MODEL
#COMMENT UNTIL model.summary() if the model is to be loaded.
model=tf.keras.Sequential()
model.add(layers.Conv2D(16, (5, 5), input_shape=(size,size,1), strides=(1,1), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (5, 5), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))  
model.add(layers.Activation('relu'))
model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(1, (1, 1), activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.summary()

#UNCOMMENT IF THE MODEL IS TO BE LOADED INSTEAD OF TRAINED
#model=load_model('model19jun12ep')


history=model.fit(data_full_fore, data_label, epochs=4, batch_size=50,
          validation_data=(val_full_fore, val_label))

#TRAINING AND VALIDATION STATISTICS

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