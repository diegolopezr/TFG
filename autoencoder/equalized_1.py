# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda, Add
from keras.optimizers import Adam
from keras import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization

BATCH_SIZE=300

def train(model, training_data):
    model.fit(training_data, training_data[0],
              epochs=9,
              batch_size=BATCH_SIZE,
              verbose=2,
              validation_split=0.1)


def filter_noise(tensor):
    n=7
    a=0.7
    b=0.5
    w_n = np.zeros((n+2,n))
    for i in range(n):    # w[n] = 1 + a*z^-1 + b*z^-2
        w_n[i][i] = b
        w_n[i+1][i] = a
        w_n[i+2][i] = 1
    w_n = tf.convert_to_tensor(w_n, dtype=tf.float32)
    result = tf.matmul(tensor, w_n)
    noise_filter_energy = (np.array([0.5, 0.7, 1], dtype=np.float32) ** 2).sum()
    result = result / np.sqrt(noise_filter_energy)
    return result


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

    EbNo_train = 10**0.8  # converted 8 dB of EbNo
    noise_input = Input(shape=(n+2,), name='Noise_input')
    gaussian = GaussianNoise(np.sqrt(1 / (2 * k * EbNo_train)))(noise_input)
    equalized_gaussian = Lambda(filter_noise, name='Equalize')(gaussian)
    encoded3 = Add()([encoded2_, equalized_gaussian])

    decoded = Dense(M, activation='relu', name='relu_2')(encoded3)
    decoded1 = Dense(M, activation='softmax', name='softmax')(decoded)

    autoencoder = Model(inputs=[input_signal, noise_input], outputs=decoded1)
    adam_opti = Adam(lr=0.0008)
    autoencoder.compile(optimizer=adam_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    print(autoencoder.summary())

    train(autoencoder, [data, np.zeros(shape=(N, n+2), dtype=np.float32)])

    encoder = Model(input_signal, encoded2_)  # Model representing the encoder part

    encoded_input = Input(shape=(n,))  # New input code sized

    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)  # Previous trained model with calculated weights representing the decoder

    return encoder, decoder
