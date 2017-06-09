def VAE(dataset, latent_dim=50, intermediate_dim=128, batch_size=100, epochs=50, epsilon_std=1.0):
    """ Variational AutoEncoder in keras.
    Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
    Author: Sun Hao <sunhopht@gmail.com>
    """
    import numpy as np
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras.callbacks import EarlyStopping
    from keras import backend as K
    from keras import objectives

    original_shape = dataset.shape
    original_dim = np.prod(original_shape[1:])

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
        xent_loss = original_dim * objectives.binary_crossentropy(x, decoder_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, decoder_mean)
    vae.compile(
        optimizer='rmsprop',
        loss=vae_loss
    )

    dataset = dataset.reshape((-1, original_dim))

    n_train = int(original_shape[0] / batch_size / 3) * batch_size
    x_train = dataset[:n_train]
    x_test  = dataset[n_train:]

    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto'
    )

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test),
            callbacks=[earlystop]
    )

    new_dataset = vae.predict(dataset, batch_size=batch_size)
    new_dataset = new_dataset.reshape(original_shape)

    return new_dataset


if __name__ == '__main__':

    import numpy as np

    data = np.random.normal(size=(10000, 100, 100))
    ndata = VAE(data)
    print(ndata.shape)
