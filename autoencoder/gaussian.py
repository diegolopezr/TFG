# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.optimizers import Adam
from keras import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization


def train(model, training_data):
    model.fit(training_data, training_data,
              epochs=9,
              batch_size=300,
              verbose=2,
              validation_split=0.1)


def create_autoencoder(n, k):
    M = 2**k
    R = k / n
    # generating data of size N
    N = 500000
    label = np.random.randint(M, size=N)

    # creating one hot encoded vectors
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)

    data = np.array(data)

    input_signal = Input(shape=(M,), name='Signal_input')
    encoded = Dense(M, activation='relu', name='relu_1')(input_signal)
    encoded1 = Dense(n, activation='linear', name='linear_1')(encoded)
    encoded2 = BatchNormalization()(encoded1)
    encoded2_ = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='Energy_normalizing')(encoded2)

    EbNo_train = 10**0.75  # converted 7.5 dB of EbNo
    encoded3 = GaussianNoise(np.sqrt(1 / (2 * k * EbNo_train)))(encoded2_)


    decoded = Dense(M, activation='relu', name='relu_2')(encoded3)
    decoded1 = Dense(M, activation='softmax', name='softmax')(decoded)

    autoencoder = Model(input_signal, decoded1)
    adam_opti = Adam(lr=0.0008)
    autoencoder.compile(optimizer=adam_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    print(autoencoder.summary())

    train(autoencoder, data)

    encoder = Model(input_signal, encoded2_)  # Model representing the encoder part

    encoded_input = Input(shape=(n,))  # New input code sized

    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)  # Previous trained model with calculated weights representing the decoder

    return encoder, decoder
