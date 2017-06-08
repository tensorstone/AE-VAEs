def VAE_(dataset=dataset,shape_0=30,shape_1=4,latent_dim = 50, intermediate_dim = 128, batchsize = 100, nb_epoch = 50, epsilon_std = 1.0):
    import numpy as np
    import os
    import pandas as pd


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    original_dim = shape_0 * shape_1 
    latent_dim = 50
    intermediate_dim =128 
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

    newdata = dataset.reshape((-1,original_dim))

    #for i in range(len(newdata)):
    #    for j in range(len(newdata[0])):
    #        newdata[i][j] = (newdata[i][j]+1)/2
    index = [i for i in range(len(dataset))]
    intnum = int(len(dataset)/batchsize-0.5)
    int_test = int(intnum /3 )*batchsize
    int_train = int_test *2
    #import random
    #random.shuffle(index)
    #newdata = newdata[index]

    x_train =(newdata[0:int_train]) 
    x_test = (newdata[int_train:int_train+int_test])
    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  

    import keras
    EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    vae.fit(x_train, x_train,  
            shuffle=True,  
            nb_epoch=nb_epoch,  
            #verbose=2,  
            batch_size=batchsize,  
            validation_data=(x_test, x_test),callbacks=[EarlyStopping])  

    import matplotlib.pyplot as plt

    #encoded_imgs = encoder.predict(x_test)
    decoded_imgs = vae.predict(x_test,batch_size=batchsize)
    size = (shape_0,shape_1)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(1,n):
        # display original
        ax = plt.subplot(2,n,i)
        plt.imshow(x_test[i].reshape(size))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n ,i + n)
        plt.imshow(decoded_imgs[i].reshape(size))
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    final_decoded_imgs = vae.predict(newdata[:int_test+int_train],batch_size=batchsize)