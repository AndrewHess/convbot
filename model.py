import tensorflow as tf
from tensorflow.keras.backend import one_hot
from tensorflow import shape

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, Reshape, Flatten
from keras.optimizers import adam

from utils.losses import generator_loss, discriminator_loss


input_size = 100
vocab_len = 5


def word_probs_to_words(x):
    ''' Determine the predicted words from the probabilites of each word.

    x: A tensor of shape (batch_size, num_words * len(vocab)) that contains the
       probability of each word at each of the num_words locations.

    Returns: A tensor of shape (batch_size, num_words, 1) by selecting the
             most probable word for each location.
    '''

    # Reshape x to (batch_size, num_words, len(vocab)). The Reshape layer does
    # not include the batch_size in the target shape.
    x = Reshape((input_size, vocab_len))(x)

    # Get the one hot mask of where the max probabilites are.
    indices = K.argmax(x, axis=-1)
    indices = K.cast(indices, dtype=tf.float32)

    return indices


def setup_model():
    # Build the models.
    gen, dis, full = build_model()

    print('discriminator model')
    dis.summary()

    # The full model is to train the generator, so freeze the discriminator.
    # full.get_layer('discriminator').trainable = False

    print('full model')
    full.summary()

    # optimizer = adam(lr=0.1)
    optimizer = adam()

    # Compile the models for training.
    dis.compile(optimizer=optimizer, loss=discriminator_loss)
    full.compile(optimizer=optimizer, loss=generator_loss)

    return gen, dis, full


def build_model():
    meaning_model = build_meaning()
    mem_input = Input(shape=(1,), name='mem_input')
    memory = Dense(2, name='memory')(mem_input)

    # Build the generator.
    gen_input = Input(shape=(input_size, vocab_len), name='gen_input')
    meaning = meaning_model(gen_input)
    concat = Concatenate(name='gen_mem_concat')([memory, meaning])

    # Create probabilites for each word for each output location.
    hidden = [Dense(vocab_len, activation='softmax', name='gen_word_' + str(i))(concat)
              for i in range(input_size)]
    concat = Concatenate(name='gen_word_concat')(hidden)
    gen_output = Reshape((input_size, vocab_len), name='gen_output')(concat)

    # Build the discriminator.
    dis_input = Input(shape=(input_size, vocab_len), name='dis_input')
    meaning = meaning_model(dis_input)
    concat = Concatenate(name='dis_concat')([memory, meaning])
    dis_output = Dense(1, activation='sigmoid', name='dis_output')(concat)

    # Setup the models.
    gen = Model(inputs=[gen_input, mem_input], outputs=gen_output, name='generator')
    dis = Model(inputs=[dis_input, mem_input], outputs=dis_output, name='discriminator')

    full_output = dis([gen_output, mem_input])
    full = Model(inputs=[gen_input, mem_input], outputs=full_output, name='full_model')

    return gen, dis, full


def build_meaning():
    ''' Build a network that determines the meaning of a sentence. '''
    input_layer = Input(shape=(input_size, vocab_len), name='meaning_input')
    hidden = Flatten(name='meaning_flatten')(input_layer)
    output_layer = Dense(10, activation='relu')(hidden)

    return Model(inputs=input_layer, outputs=output_layer, name='meaning')
