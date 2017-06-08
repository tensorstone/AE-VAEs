
# coding: utf-8

# In[648]:

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


# In[696]:

np.shape(cleardata)
#useful_data = np.asarray(cleardata)[:,560:1600]
useful_data = np.asarray(cleardata)[:,:80]


# In[697]:


usefuldataT = useful_data.T


# In[698]:

usefuldataT.shape


# In[699]:

import os
import glob
from PIL import Image
import numpy as np
imgs = 0
#NUB = 1040
NUB = 80
Size = 32
def load_data():
    NUB =80
    data = np.empty((NUB,Size,Size))
    #label = np.empty((9500,),dtype="uint8")
    for i in range(NUB):
        #imgs= "/home/ai-i-sunhao/Stim/StimImg_Shape/Shapeimg_" + str(i+1) +".jpg"
        img = Image.open("/home/ai-i-sunhao/Stim/StimImg_Shape/Shapeimg_" + str(i+1) +".jpg")
        arr = np.asarray(img)
        dst=transform.resize(arr[:,:,0], (Size, Size))
        data[i,:,:] = dst #arr_t
    data =data*1.0
    #data =data- np.mean(data)
    return data


# In[700]:

data_img = load_data()
data_img.shape


# In[701]:

data_img[0]


# In[702]:

data_img = data_img.reshape(( len(data_img),np.prod((Size,Size))))


# In[703]:

data_img.shape


# In[704]:

import random
shuf_ind = [i for i in range(NUB)]
random.shuffle(shuf_ind)
print(shuf_ind)


# In[705]:

data_img = data_img[shuf_ind]
usefuldataT = usefuldataT[shuf_ind]


# In[706]:

train_img = data_img[:70]
test_img = data_img[70:]


# In[707]:

train_img.shape


# In[708]:

useful_data.T.shape


# In[813]:

#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import regularizers
import tensorflow as tf
# this is our input placeholder
input_img = Input(shape=(1137,))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
encoded = Dense(160, activation='sigmoid')(input_img)
#encoded = tf.reshape(encoded,[-1,10,10,50])
encoded = Reshape((4, 4,10))(encoded)

x = Conv2D(10, (3, 3), activation='relu', padding='same',kernel_initializer='normal',activity_regularizer=regularizers.l1(10e-6))(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(20, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(40,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#decoded = tf.reshape(decoded,[-1,160*160])

autoencoder = Model(inputs=input_img, outputs=decoded)


# In[814]:

decoded.shape


# In[815]:

autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')


# In[ ]:




# In[816]:

x_train = np.reshape(train_img, (len(train_img), Size, Size,1))
x_test = np.reshape(test_img, (len(test_img), Size, Size,1))


# In[821]:

autoencoder.fit(useful_data.T[:70], x_train,
                nb_epoch=10,
                batch_size=10,
                shuffle=True,
                validation_data=(useful_data.T[70:], x_test))


# In[822]:

# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
Size = 32
#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(useful_data.T[70:])


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(Size, Size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(Size,Size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[260]:

x_train = np.reshape(train_img, (len(train_img), 32, 32,1))
x_test = np.reshape(test_img, (len(test_img), 32, 32,1))


# In[725]:

input_img = Input(shape=(32,32,1))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
x = Conv2D(40, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(input_img)
x = MaxPooling2D((2, 2),  padding='same')(x)

x = Conv2D(20, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(x)
encoded = MaxPooling2D((2, 2),  padding='same')(x)

x = Conv2D(10, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(40,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#decoded = tf.reshape(decoded,[-1,160*160])

autoencoder = Model(inputs=input_img, outputs=decoded)


# In[726]:

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[727]:

x_train.shape


# In[728]:

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[729]:

# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(32,32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[217]:

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),  28, 28,1))
x_test = np.reshape(x_test, (len(x_test), 28, 28,1))


# In[218]:

x_train = x_train[:50]


# In[219]:

x_train.shape


# In[ ]:




# In[220]:

input_img = Input(shape=(28,28,1))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
x = Conv2D(40, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(input_img)
x = MaxPooling2D((2, 2),  padding='same')(x)

x = Conv2D(20, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(x)
encoded = MaxPooling2D((2, 2),  padding='same')(x)

x = Conv2D(10, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(40,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#decoded = tf.reshape(decoded,[-1,160*160])

autoencoder = Model(inputs=input_img, outputs=decoded)


# In[221]:

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[222]:

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[223]:

# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:



