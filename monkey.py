
# coding: utf-8

# In[1]:

import numpy as np
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

Dataframe = pd.read_csv('/home/ai-i-sunhao/Stim/ShapeRespMatrix.txt',header=None,delim_whitespace=True)

Counter = 0
for i in range(Dataframe.shape[0]):
    if np.isnan(np.min(Dataframe.values[i]))==True:
        Counter+=1
        print(i)
print(Counter)

Counter = 0
for i in range(Dataframe.shape[0]):
    if np.isnan(np.max(Dataframe.values[i]))==True:
        Counter+=1
        print(i)
print(Counter)

cleardata = []
for i in range(len(Dataframe.values)):
    if np.isnan(np.max(Dataframe.values[i]))==False:
        cleardata.append(Dataframe.values[i])

np.shape(cleardata)

#useful
label_shape=[]
shape_label=[]
for i in range(1,81):
    label_shape.append("ell")
    shape_label.append(0)
for i in range(81,241):
    label_shape.append("longline_1")
    shape_label.append(1)
for i in range(241,321):
    label_shape.append("guangshan_1")
    shape_label.append(2)
for i in range(321,561):
    label_shape.append("shortline_1")
    shape_label.append(3)
    
for i in range(561,641):
    label_shape.append("halfline")
    shape_label.append(4)
for i in range(641,1601):
    label_shape.append("thick_halfline")
    shape_label.append(4)
for i in range(1601,2601):
    label_shape.append("triangle")
    shape_label.append(5)    
for i in range(3921,4921):
    label_shape.append("upangle")
    shape_label.append(6)
for i in range(4921,5921):
    label_shape.append("downangle")
    shape_label.append(7)
    
    
Max_ = 2600


# In[239]:


    
# for i in range(321,561):
#     label_shape.append("shortline_1")
    
# for i in range(561,1601):
#     label_shape.append("line")    
# for i in range(1601,3601):
#     label_shape.append("triangle")
    
    
# for i in range(3601,3921):
#     label_shape.append("part_cir")    
# for i in range(3921,6241):
#     label_shape.append("angle")    
# for i in range(6241,6471):
#     label_shape.append("star")    
# for i in range(6471,6551):
#     label_shape.append("Y")    
# for i in range(6551,6631):
#     label_shape.append("shanxing")
    
# for i in range(6631,1601):
#     label_shape.append("line")
        
# for i in range(561,1601):
#     label_shape.append("line")


# In[251]:

len(shape_label)
shape_label = np.asarray(shape_label)


# In[252]:

activation_info=np.vstack((np.mat(cleardata)[:,:Max_].T,np.mat(cleardata)[:,3920:5920].T))# shallow copy
#activation_info=np.mat(cleardata)[:,3920:5920].T


# In[253]:

activation_info.shape


# In[254]:

shape_label.shape


# In[255]:

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#from data import load_data
import random
import numpy as np

np.random.seed(1024)  # for reproducibility


# In[245]:

Catag= 8

data = activation_info
label = shape_label
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')
label = np_utils.to_categorical(label, Catag)


# In[246]:

data = data[:,:]


# In[247]:

data.shape


# In[248]:

label.shape


# In[249]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()


#model.add(Conv2D(1,(1,1),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(1,1137))) 
#model.add(Flatten(input_shape=(1137,1)))
model.add(Dense( 500, init='normal',input_shape=(1137,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense( 500, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(Catag,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

#tensorboard = TensorBoard(log_dir='./monkeylogs/run_1', histogram_freq=0)
#checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=30, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2)#, callbacks=[tensorboard])


# In[ ]:




# In[ ]:




# In[27]:

import os
import glob
from PIL import Image
import numpy as np
imgs = 0
def load_data():
    data = np.empty((9500,160,160,3),dtype="float32")
    #label = np.empty((9500,),dtype="uint8")
    for i in range(9500):
        #imgs= "/home/ai-i-sunhao/Stim/StimImg_Shape/Shapeimg_" + str(i+1) +".jpg"
        img = Image.open("/home/ai-i-sunhao/Stim/StimImg_Shape/Shapeimg_" + str(i+1) +".jpg")
        arr = np.asarray(img,dtype = "float32")
        data[i,:,:,:] = arr #arr_t
    data =data/255
    #data =data- np.mean(data)
    return data


# In[28]:

data111 = load_data()


# In[29]:

lebel = data_1


# In[30]:


from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#from data import load_data
import random
import numpy as np

np.random.seed(1024)  # for reproducibility

#加载数据
data,label = data111,data_1

#打乱数据


index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 2)

###############kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)
#开始建立CNN模型
###############


# In[31]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(10,10),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(160,160,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (10, 10),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./monkeylogs/run_1', histogram_freq=0)
#checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[tensorboard])


# In[34]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 3,(1,1),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(160,160,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Flatten())
model.add(Dense(500, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(500, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./monkeylogs/run_1', histogram_freq=0)
#checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[tensorboard])


# In[ ]:



