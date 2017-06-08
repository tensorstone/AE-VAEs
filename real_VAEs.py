
# coding: utf-8

# In[1]:

import numpy as np
import os
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[2]:

'''''This script demonstrates how to build a variational autoencoder with Keras. 
 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  
  

batch_size = 100  
original_dim = 784   
latent_dim = 5  
intermediate_dim = 256  
nb_epoch = 50  
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))  
h = Dense(intermediate_dim, activation='relu')(x)  
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


def vae_loss(x, decoder_mean):  
    xent_loss = original_dim * objectives.binary_crossentropy(x,decoder_mean)  
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + kl_loss  
  
vae = Model(x, decoder_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  
  


(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.pkl.gz')  
  
x_train = x_train.astype('float32') / 255.  
x_test = x_test.astype('float32') / 255.  
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  
  



# In[3]:

vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test))  


# In[5]:

# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test,batch_size=100)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[24]:

type(x_test)


# In[16]:

z_mean.shape


# In[6]:

# build a model to project inputs on the latent space  
encoder = Model(x, z_mean)
  
# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)  
plt.colorbar()  
plt.show()  
  


# In[ ]:


# build a digit generator that can sample from the learned distribution  
decoder_input = Input(shape=(latent_dim,))  
_h_decoded = decoder_h(decoder_input)  
_x_decoded_mean = decoder_mean(_h_decoded)  
generator = Model(decoder_input, _x_decoded_mean)  
  
# display a 2D manifold of the digits  
n = 15  # figure with 15x15 digits  
digit_size = 28  
figure = np.zeros((digit_size * n, digit_size * n))  
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian  
# to produce values of the latent variables z, since the prior of the latent space is Gaussian  
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))  
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))  
  
for i, yi in enumerate(grid_x):  
    for j, xi in enumerate(grid_y):  
        z_sample = np.array([[xi, yi]])  
        x_decoded = generator.predict(z_sample)  
        digit = x_decoded[0].reshape(digit_size, digit_size)  
        figure[i * digit_size: (i + 1) * digit_size,  
               j * digit_size: (j + 1) * digit_size] = digit
        
plt.figure(figsize=(10, 10))  
plt.imshow(figure, c='red')
plt.colorbar()
plt.show()  
  
plot(vae,to_file='variational_autoencoder_vae.png',show_shapes=True)  
plot(encoder,to_file='variational_autoencoder_encoder.png',show_shapes=True)  
plot(generator,to_file='variational_autoencoder_generator.png',show_shapes=True)  


# In[3]:

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(9,6))
n=1000
#rand 均匀分布和 randn高斯分布
x=np.random.randn(1,n)
y=np.random.randn(1,n)
T=np.arctan2(x,y)
plt.scatter(x,y,c=T,s=25,alpha=0.4,marker='o')
#T:散点的颜色
#s：散点的大小
#alpha:是透明程度
plt.show()


# In[10]:




# In[ ]:




# In[8]:

newdata = pd.read_csv("IsingModel16_16simul.csv").values


# In[9]:

newdata.shape


# In[ ]:




# In[ ]:




# In[84]:

'''''This script demonstrates how to build a variational autoencoder with Keras. 
 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  
  

batch_size = 100  
original_dim = 256   
latent_dim = 4
intermediate_dim = 256  
nb_epoch = 50  
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))  
h = Dense(intermediate_dim, activation='relu')(x)  
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


def vae_loss(x, decoder_mean):  
    xent_loss = original_dim * objectives.binary_crossentropy(x,decoder_mean)  
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + kl_loss  
  
vae = Model(x, decoder_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  

newdata = pd.read_csv("IsingModel16_16simul.csv").values
for i in range(len(newdata)):
    for j in range(len(newdata[0])):
        newdata[i][j] = (newdata[i][j]+1)/2
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
x_train = newdata[:15000]
x_test = newdata[15000:20000]
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  




# In[95]:

vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test))  


# In[96]:

# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test,batch_size=100)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(16, 16))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(16,16))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[130]:

T = np.zeros((20099,))
u=5.025
for i in range(20099):
    
    if i%101==0:
        u = u- 0.025
    T[i]=u
T = T[index]

T_train = T[:15000]
T_test = T[15000:20000]


# In[131]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= np.sum(x_test[i])

encoder = Model(x, z_mean)
  
# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
plt.scatter( x_test_encoded[:, 1] ,y_test , c=T_test)  
plt.colorbar()  
plt.show()  
  


# In[43]:


y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test[i] = np.sum(x_test)


# In[45]:

y_test.shape


# In[37]:

newdata = pd.read_csv("IsingModel16_16simul.csv").values
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
x_train = newdata[:15000]
x_test = newdata[15000:]
#x_train = x_train.reshape((-1,16,16,1))
#x_test = x_test.reshape((-1,16,16,1))


# In[41]:

x_train.shape


# In[42]:

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import regularizers
import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  

input_img = Input(shape=(256,))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
encoded =Dense(100,activation='relu',kernel_initializer='normal')(input_img)
#x = MaxPooling2D((2, 2),  padding='same')(x)

x = Dense(100,  activation='relu',kernel_initializer='normal')(encoded)
decoded = Dense(256,activation='sigmoid')(x)

#decoded = tf.reshape(decoded,[-1,160*160])

autoencoder = Model(inputs=input_img, outputs=decoded)


# In[43]:

autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')


# In[44]:

autoencoder.fit(x_train, x_train,
                nb_epoch=10,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[45]:

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
    plt.imshow(x_test[i].reshape(16, 16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(16,16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[46]:


y_test = np.zeros((1,len(x_test)))
for i in range(len(x_test)):
    y_test = np.sum(x_test)
# display a 2D plot of the digit classes in the latent space  
input_img = Input(shape=(256,))

encoded =Dense(100,activation='relu',kernel_initializer='normal')(input_img)
#encoded = MaxPooling2D((2, 2),  padding='same')(x)
encoder = Model(x, encoded)



x_test_encoded = encoder.predict(x_test, batch_size=10)  
plt.figure(figsize=(6, 6))  
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)  
plt.colorbar()  
plt.show()  


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[182]:

'''''This script demonstrates how to build a variational autoencoder with Keras. 
 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  
  

batch_size = 100  
original_dim = 1024  
latent_dim = 1
intermediate_dim = 256  
nb_epoch = 50  
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))  
h = Dense(intermediate_dim, activation='relu')(x)  
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


def vae_loss(x, decoder_mean):  
    xent_loss = original_dim * objectives.binary_crossentropy(x,decoder_mean)  
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + kl_loss  
  
vae = Model(x, decoder_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  

newdata = pd.read_csv("IsingModel32_32simul.csv").values
T = pd.read_csv("Temp32_32.csv").values
T = np.asarray(np.reshape(T,[-1,len(T)]))
T = T[0]

for i in range(len(newdata)):
    for j in range(len(newdata[0])):
        newdata[i][j] = (newdata[i][j]+1)/2
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
T = T[index]
x_train = newdata[:5000]
x_test = newdata[5000:9500]
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  




# In[183]:

vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test))  


# In[184]:


# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test,batch_size=100)
size = 32

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(size, size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(size,size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[185]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[5000:9500]

encoder = Model(x, z_mean)
  
# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[144]:




# In[146]:

T


# In[21]:

'''''This script demonstrates how to build a variational autoencoder with Keras. 
 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
from keras.datasets import mnist  
from keras.utils.vis_utils import plot_model  
import sys  
  

batch_size = 100  
original_dim = 1024  
latent_dim = 1
intermediate_dim = 256  
nb_epoch = 50  
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))  
h = Dense(intermediate_dim, activation='relu')(x)  
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


def vae_loss(x, decoder_mean):  
    xent_loss = original_dim * objectives.binary_crossentropy(x,decoder_mean)  
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + kl_loss  
  
vae = Model(x, decoder_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  

newdata = pd.read_csv("IsingModel32_32simul_6.csv").values
T = pd.read_csv("Temp32_32_6.csv").values
T = np.asarray(np.reshape(T,[-1,len(T)]))
T = T[0]

for i in range(len(newdata)):
    for j in range(len(newdata[0])):
        newdata[i][j] = (newdata[i][j]+1)/2
index = [i for i in range(len(newdata))]
import random
random.shuffle(index)
newdata = newdata[index]
T = T[index]
x_train = newdata[:5000]
x_test = newdata[5000:9500]
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  




# In[22]:

vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        #verbose=2,  
        batch_size=100,  
        validation_data=(x_test, x_test))  
# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)


# In[20]:

import matplotlib.pyplot as plt

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test,batch_size=100)
size = 32

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(size, size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n ,i + n)
    plt.imshow(decoded_imgs[i].reshape(size,size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[23]:


y_test = np.zeros((len(x_test),))
for i in range(len(x_test)):
    y_test[i]= (np.sum(x_test[i])-512)*2

T_test = T[5000:9500]

encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
plt.scatter( x_test_encoded[:, 0] ,y_test , c=T_test)  
plt.colorbar()
plt.show()  
  


# In[ ]:



