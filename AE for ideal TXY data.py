
# coding: utf-8

# In[1]:

import numpy as np
import os
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[2]:

Sample_Num = 5000
start = []
for i in range(Sample_Num ):
    start.append(2*np.pi * np.random.random()-np.pi)
    
    
deg_120 = np.pi*120/180
lattice = [] 
for i in range(Sample_Num ):
    line_1 = []
    line_2 = []
    line_3 = []
    for j in range(15):
        line_1.append(start[i]+j*deg_120) 
    for j in range(15):
        line_2.append(start[i]+j*deg_120 + deg_120)
    for j in range(15):
        line_3.append(start[i]+j*deg_120 + 2 * deg_120)
    lattice.append(np.reshape(np.vstack((line_1,line_2,line_3)*5),[225]))

    
    
lattice = np.asarray(lattice)
lat_cos = np.zeros((Sample_Num ,225))
lat_sin = np.zeros((Sample_Num ,225))
for i in range(Sample_Num ):
    lat_cos[i]= np.cos(lattice[i])
    lat_sin[i]= np.sin(lattice[i])
test_artificial = np.hstack((lat_cos,lat_sin))

start = []
for i in range(Sample_Num ):
    start.append(2*np.pi * np.random.random()-np.pi)    
    
deg_120 = - np.pi*120/180
lattice_2 = [] 
for i in range(Sample_Num ):
    line_1 = []
    line_2 = []
    line_3 = []
    for j in range(15):
        line_1.append(start[i]+j*deg_120) 
    for j in range(15):
        line_2.append(start[i]+j*deg_120 + deg_120)
    for j in range(15):
        line_3.append(start[i]+j*deg_120 + 2 * deg_120)
    lattice_2.append(np.reshape(np.vstack((line_1,line_2,line_3)*5),[225]))
    
lat_cos_2 = np.zeros((Sample_Num ,225))
lat_sin_2 = np.zeros((Sample_Num ,225))
for i in range(Sample_Num ):
    lat_cos_2[i]= np.cos(lattice_2[i])
    lat_sin_2[i]= np.sin(lattice_2[i])
    
test_artificial_2 = np.hstack((lat_cos_2,lat_sin_2))
artificial_data = np.vstack((test_artificial,test_artificial_2))


# In[3]:

artificial_data.shape


# In[126]:

from keras.regularizers import l2
from keras import regularizers
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
batch_size = 100  
original_dim = 450 
row = 15
col = 15
channel = 2
latent_dim = 4
intermediate_dim = 100
nb_epoch = 100  
epsilon_std = 1.0  



input_img = Input(shape=(450,))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
encoded = Dense(latent_dim,kernel_initializer='normal',kernel_regularizer=regularizers.l2(0.001))(input_img)


decoded =Dense(450, activation='sigmoid')(encoded)
#decoded = tf.reshape(decoded,[-1,160*160])

autoencoder = Model(inputs=input_img, outputs=decoded)


# In[127]:



newdata = artificial_data
T = np.zeros((1,len(newdata)))

#for i in range(len(newdata)):
#    for j in range(len(newdata[0])):
#        newdata[i][j] = (newdata[i][j]+1)/2
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
T = T[0]
T = T[index]

    
newdata = newdata[index]
T = T[index]


x_train = newdata[:8000]
x_test = newdata[8000:]

import keras
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')  
autoencoder.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test),callbacks=[EarlyStopping])  
# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)


# In[128]:

import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs =autoencoder.predict(x_test,batch_size=100)
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


# In[129]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 1] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[130]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 2] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[131]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 0] ,x_test_encoded[:, 3] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[132]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 1] ,x_test_encoded[:, 2] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[133]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 1] ,x_test_encoded[:, 3] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[134]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[8000:]

encoder = Model(input_img, encoded)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
#plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.scatter( x_test_encoded[:, 2] ,x_test_encoded[:, 1] , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[135]:

W0 = np.sqrt(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[450,1]))[0][0])
W1 = np.sqrt(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[450,1]))[0][0])
W2 = np.sqrt(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[450,1]))[0][0])
W3 = np.sqrt(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))[0][0])
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[450,1]))/W0/W1)
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[450,1]))/W0/W2)
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W0/W3)
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[450,1]))/W1/W2)
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W1/W3)
print(np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W2/W3)


# In[136]:

A = [np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[450,1]))/W0/W1,
np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[450,1]))/W0/W2,
np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[0]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W0/W3,
np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[450,1]))/W1/W2,
np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[1]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W1/W3,
np.matmul(np.reshape((autoencoder.layers[1].get_weights()[0].T[2]),[1,450]),np.reshape((autoencoder.layers[1].get_weights()[0].T[3]),[450,1]))/W2/W3]


# In[137]:

np.std(A)


# In[138]:

sum(np.abs(A))


# In[139]:

autoencoder.layers[1].get_weights()[0].shape


# In[140]:

Max = np.max(np.matmul(autoencoder.layers[1].get_weights()[0].T,autoencoder.layers[1].get_weights()[0]))


# In[141]:

np.matmul(autoencoder.layers[1].get_weights()[0].T,autoencoder.layers[1].get_weights()[0])/Max#l2 = 0.001


# In[ ]:




# In[ ]:




# In[124]:

np.matmul(autoencoder.layers[1].get_weights()[0].T,autoencoder.layers[1].get_weights()[0])/Max#l2 = 0.01


# In[125]:

plt.imshow(autoencoder.layers[1].get_weights()[0].T[:,:10])
plt.colorbar()
plt.show()
plt.imshow(autoencoder.layers[1].get_weights()[0].T[:,10:20])
plt.colorbar()
plt.show()


# In[ ]:



