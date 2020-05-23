# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda, Concatenate
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


def filter_signal(tensor):
    n = 7
    filter = np.zeros((2*n,n))
    FIR_coeficients = np.array([-0.0799, 0.1262, -0.0168, -0.2289, 0.3540, -0.0379, -0.655, 0.9929], dtype=np.float32)
    for i in range(n):
        filter[i][i] = FIR_coeficients[0]
        filter[i+1][i] = FIR_coeficients[1]
        filter[i+2][i] = FIR_coeficients[2]
        filter[i+3][i] = FIR_coeficients[3]
        filter[i+4][i] = FIR_coeficients[4]
        filter[i+5][i] = FIR_coeficients[5]
        filter[i+6][i] = FIR_coeficients[6]
        filter[i+7][i] = FIR_coeficients[7]
    filter = tf.convert_to_tensor(filter, dtype=tf.float32)
    output = tf.matmul(tensor, filter)
    signal_filter_energy = (np.array([-0.0799, 0.1262, -0.0168, -0.2289, 0.3540, -0.0379, -0.655, 0.9929], dtype=np.float32) ** 2).sum()
    output = output / np.sqrt(signal_filter_energy)
    return output


def create_autoencoder(n, k):
    M = 2**k
    R = k / n
    # generating data of size N
    N = 500000
    label = np.random.randint(M, size=N)

    # creating two hot encoded vectors
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)

    data = np.array(data)
    data_delayed = np.concatenate((np.zeros((1,M)),data[:-1]))   # Adelantados una posicion

    input_signal = Input(shape=(M,), name='Signal_input')
    input_signal_delayed = Input(shape=(M,), name='Signal_delayed_input')
    encoded = Dense(M, activation='relu', name='relu_1')(input_signal)
    encoded_delayed = Dense(M, activation='relu', name='relu_delayed')(input_signal_delayed)
    encoded1 = Dense(n, activation='linear', name='linear_1')(encoded)
    encoded1_delayed = Dense(n, activation='linear', name='linear_delayed_1')(encoded_delayed)
    encoded2 = BatchNormalization()(encoded1)
    encoded2_ = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='Energy_normalizing')(encoded2)
    encoded2_delayed = BatchNormalization()(encoded1_delayed)
    encoded2_delayed_ = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='Energy_normalizing_2')(encoded2_delayed)

    concatenated_data = Concatenate(axis=1, name='Concatenate')([encoded2_delayed_, encoded2_])
    filtered_signal = Lambda(filter_signal, name='signal_equalizer')(concatenated_data)

    EbNo_train = 10**0.8  # converted 8 dB of EbNo
    encoded3 = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(filtered_signal)

    decoded = Dense(M, activation='relu', name='relu_2')(encoded3)
    decoded1 = Dense(M, activation='softmax', name='softmax')(decoded)

    autoencoder = Model(inputs=[input_signal, input_signal_delayed], outputs=decoded1)
    adam_opti = Adam(lr=0.0008)
    autoencoder.compile(optimizer=adam_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    print(autoencoder.summary())

    train(autoencoder, [data, data_delayed])

    encoder = Model(input_signal, encoded2_)  # Model representing the encoder part

    encoded_input = Input(shape=(n,))  # New input code sized

    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)  # Previous trained model with calculated weights representing the decoder

    return encoder, decoder
