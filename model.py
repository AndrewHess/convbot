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


def setup_model():
    ''' Build and compile the models. '''

    # Build the models.
    gen, dis, full = build_model()

    # The full model is to train the generator, so freeze the discriminator.
    full.get_layer('discriminator').trainable = False

    # Make it so that only the discriminator can learn sentence meaning.
    full.get_layer('gen_meaning').trainable = False

    # print('full model')
    # full.summary()
    # print('discriminator model')
    # dis.summary()

    # Compile the models for training.
    dis.compile(optimizer='adam', loss=discriminator_loss)
    full.compile(optimizer='adam', loss=generator_loss)

    return gen, dis, full


def build_model():
    ''' Build the generator, discriminator, and combined models. '''

    # Setup the memory.
    mem_input = Input(shape=(1,), name='mem_input')
    gen_memory = Dense(2, name='gen_memory')(mem_input)
    dis_memory = Dense(2, name='dis_memory')(mem_input)

    # Build the generator.
    gen_input = Input(shape=(input_size, vocab_len), name='gen_input')
    gen_meaning = build_meaning(True)(gen_input)
    gen_mem_concat = Concatenate(name='gen_mem_concat')([gen_meaning, gen_memory])

    # Create probabilites for each word for each output location.
    hidden = [Dense(vocab_len, activation='softmax', name='gen_word_' + str(i))(gen_mem_concat)
              for i in range(input_size)]
    gen_word_concat = Concatenate(name='gen_word_concat')(hidden)
    gen_output = Reshape((input_size, vocab_len), name='gen_output')(gen_word_concat)

    # Build the discriminator.
    dis_meaning_model = build_meaning(False)
    dis_input = Input(shape=(input_size, vocab_len), name='dis_input')
    dis_meaning = dis_meaning_model(dis_input)
    dis_gen_meaning = dis_meaning_model(gen_input)
    dis_concat = Concatenate(name='dis_concat')([dis_gen_meaning, dis_meaning, dis_memory])
    dis_output = Dense(1, activation='sigmoid', name='dis_output')(dis_concat)

    # Setup the models.
    gen = Model(inputs=[gen_input, mem_input], outputs=gen_output, name='generator')
    dis = Model(inputs=[gen_input, dis_input, mem_input], outputs=dis_output, name='discriminator')

    full_output = dis([gen_input, gen_output, mem_input])
    full = Model(inputs=[gen_input, mem_input], outputs=full_output, name='full_model')

    return gen, dis, full


def build_meaning(is_gen):
    ''' Build a network that determines the meaning of a sentence. '''

    model_name = 'gen_meaning' if is_gen else 'dis_meaning'

    input_layer = Input(shape=(input_size, vocab_len), name='meaning_input')
    hidden = Flatten(name='meaning_flatten')(input_layer)
    output_layer = Dense(10, activation='relu')(hidden)

    return Model(inputs=input_layer, outputs=output_layer, name=model_name)
