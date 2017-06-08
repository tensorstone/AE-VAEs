
# coding: utf-8

# In[337]:

import numpy as np
import os
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[339]:

'''''This script demonstrates how to build a variational autoencoder with Keras. 
 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda,Conv2D, MaxPooling2D,Flatten, Reshape
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  
#import tensorflow as tf

batch_size = 100  
original_dim = 450 
row = 15
col = 15
channel = 2
latent_dim = 2
intermediate_dim = 100
nb_epoch = 100  
epsilon_std = 1.0  

#my tips:encoding  
x = Input(shape=(original_dim,))
x_res = Reshape([row,col,channel])(x)
conv1 = Conv2D(32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),activation='relu')(x_res)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default')(conv1)
#conv2 = Conv2D(32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),activation='relu')(pool1)
#pool2 = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default')(conv2)
conv_flat = Flatten()(pool1)
h = Dense(intermediate_dim, activation='relu')(conv_flat)  
z_mean = Dense(latent_dim)(h)  
z_log_var = Dense(latent_dim)(h)  

#my tips:Gauss sampling,sample Z  
def sampling(args):   
    z_mean, z_log_var = args  
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,stddev=epsilon_std)  
    return z_mean + K.exp(z_log_var / 2) * epsilon  
  
# note that "output_shape" isn't necessary with the TensorFlow backend  
# my tips:get sample z(encoded)  
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# we instantiate these layers separately so as to reuse them later  
decoder_h = Dense(intermediate_dim, activation='relu')(z) 
decoder_mean = Dense(original_dim, activation='sigmoid')(decoder_h)

#x_res = np.reshape(x,(-1,450))

def vae_loss(x, decoder_mean):  
    xent_loss = original_dim * objectives.binary_crossentropy(x,decoder_mean)  
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + 19*kl_loss  
  
vae = Model(x, decoder_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  

newdata = pd.read_csv("tri-XYModel15_15simul.csv").values[14999:]
T = pd.read_csv("tri-Temp15_15.csv").values[14999:]
T = np.asarray(np.reshape(T,[-1,len(T)]))
T = T[0]

#for i in range(len(newdata)):
#    for j in range(len(newdata[0])):
#        newdata[i][j] = (newdata[i][j]+1)/2
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
T = T[index]
x_train_1 =np.cos(newdata[0:3000]) 
x_test_1 = np.cos(newdata[3000:5100])
x_train_2 = np.sin(newdata[0:3000])
x_test_2 = np.sin(newdata[3000:5100])
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  



# In[340]:

x_train = np.hstack((x_train_1,x_train_2))
x_test = np.hstack((x_test_1,x_test_2))
#x_train = x_train.reshape((-1,row,col,channel))
#x_test = x_test.reshape((-1,row,col,channel))


# In[341]:

import keras
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')


vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test),callbacks=[EarlyStopping])  
# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)


# In[342]:

import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test,batch_size=100)
size = 15

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i][:225].reshape(size, size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i][:225].reshape(size,size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[343]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 1] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[344]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 2] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 3] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 1] ,x_test_encoded[:, 2] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 1] ,x_test_encoded[:, 3] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[3000:5100]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 2] ,x_test_encoded[:, 3] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:




# In[ ]:



