# -*- coding: utf-8 -*-

"""
This script loads the training images, compiles a CNN architecture and trains it.

Version: September 18, 2019
@author David Balbas
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from pathlib import Path
parent_path=os.fspath(Path(__file__).parent.parent)

data_path=parent_path+'/7mindata/npymats'

#Image reading

size=150
nmaps=2 #number of full maps to train the net. 2 is recommended (roughly 6150 images)
data_full=[]
data_label=[]
val_full=[]
val_label=[]

lenj=2900*nmaps #length of the training set. 3072*nmaps-lenj=length of the validation set.
indices=np.arange(3072*nmaps)
np.random.shuffle(indices)
dataindices=indices[:lenj]
valindices=indices[lenj:]


for j in dataindices:
    i=1+j//3072
    rem=j%3072
    full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
    data_full.append(np.array(full))
    label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
    data_label.append(np.array(label))

        
for j in valindices:
    i=1+j//3072
    rem=j%3072
    full=np.load(data_path+'/full_'+str(i)+'_'+str(rem)+'.npy')
    val_full.append(np.array(full))
    label=np.load(data_path+'/segps_'+str(i)+'_'+str(rem)+'.npy')
    val_label.append(np.array(label))


#Up to this point: matrices loaded in data_(full,label,mfiltered), val_().
#Reshaping of lists to convert them into input tensors
data_full=np.array(data_full).reshape(lenj,size,size,1)
data_label=np.array(data_label).reshape(lenj,size,size,1)
val_full=np.array(val_full).reshape(3072*nmaps-lenj,size,size,1)
val_label=np.array(val_label).reshape(3072*nmaps-lenj,size,size,1)

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

history1=model.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model.save('cnn7min4ep')
#TRAINING AND VALIDATION STATISTICS

history_dict=history1.history
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


history2=model.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model.save('cnn7min8ep')
#TRAINING AND VALIDATION STATISTICS

history_dict=history2.history
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

history3=model.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model.save('cnn7min12ep')

history_dict=history3.history
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
plt.savefig('loss.svg', format='svg')

plt.show()
plt.close()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.svg', format='svg')

plt.show()

model2=tf.keras.Sequential()
model2.add(layers.Conv2D(16, (7, 7), input_shape=(size,size,1), strides=(1,1), padding='same'))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.Conv2D(32, (7, 7), padding='same'))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))  
model2.add(layers.Activation('relu'))
model2.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
model2.add(layers.BatchNormalization())
model2.add(layers.Activation('relu'))
model2.add(layers.Conv2D(1, (1, 1), activation='sigmoid'))

model2.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model2.summary()

#UNCOMMENT IF THE MODEL IS TO BE LOADED INSTEAD OF TRAINED
#model=load_model('model19jun12ep')

dhistory1=model2.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model2.save('cnn2_7min4ep')
#TRAINING AND VALIDATION STATISTICS

history_dict=dhistory1.history
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


dhistory2=model2.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model2.save('cnn2_7min8ep')
#TRAINING AND VALIDATION STATISTICS

history_dict=dhistory2.history
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

dhistory3=model2.fit(data_full, data_label, epochs=4, batch_size=50,
          validation_data=(val_full, val_label))

model2.save('cnn2_7min12ep')

history_dict=dhistory3.history
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
plt.savefig('loss.svg', format='svg')

plt.show()
plt.close()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.svg', format='svg')

plt.show()
