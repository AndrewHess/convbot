import tensorflow as tf
from tensorflow.keras.backend import one_hot
from tensorflow import shape

import keras.backend as K
from keras.models import Model, clone_model
from keras.layers import Input, Dense, Concatenate, Lambda, Reshape, Flatten
from keras.optimizers import adam

from utils.losses import generator_loss, discriminator_loss


input_size = 10
vocab_len = 20


def setup_model():
    ''' Build and compile the models. '''

    # Build the models.
    gen, dis, full = build_model()

    # The full model is to train the generator, so freeze the discriminator.
    # Due to a bug in preserving the trainable state of a sub model after
    # saving and loading a model, each layer has to be set to untrainable
    # instead of just setting the discriminator sub model to be untrainable.
    for layer in full.get_layer('discriminator').layers:
        layer.trainable = False

    full.get_layer('discriminator').set_weights(dis.get_weights())

    # Compile the models for training.
    dis.compile(optimizer='adam', loss=discriminator_loss)
    full.compile(optimizer='adam', loss=generator_loss)

    # Show the model architectures.
    print('discriminator')
    dis.summary()

    print('stacked')
    full.summary()

    return gen, dis, full


def build_model():
    ''' Build the generator, discriminator, and combined models. '''

    # Setup the memory.
    mem_input = Input(shape=(1,), name='mem_input')
    gen_memory = Dense(2, name='gen_memory')(mem_input)
    dis_memory = Dense(2, name='dis_memory')(mem_input)

    # Build the generator.
    gen_input = Input(shape=(input_size, vocab_len), name='gen_input')
    encoded = Dense(10, name='gen_encoded')(gen_input)
    flatten = Flatten()(encoded)
    gen_meaning = Dense(10, name='gen_meaning')(flatten)
    gen_mem_concat = Concatenate(name='gen_mem_concat')([gen_meaning, gen_memory])

    # Create probabilites for each word for each output location.
    hidden = [Dense(vocab_len, activation='softmax', name='gen_word_' + str(i))(gen_mem_concat)
              for i in range(input_size)]
    gen_word_concat = Concatenate(name='gen_word_concat')(hidden)
    gen_output = Reshape((input_size, vocab_len), name='gen_output')(gen_word_concat)

    # Build the discriminator.
    dis_input = Input(shape=(input_size, vocab_len), name='dis_input')
    encoded_dis = Dense(10, name='dis_encoded')(dis_input)
    encoded_gen = Dense(10, name='dis_gen_encoded')(gen_input)
    flatten_dis = Flatten()(encoded_dis)
    flatten_gen = Flatten()(encoded_gen)
    dis_meaning = Dense(10, name='dis_meaning')(flatten_dis)
    dis_gen_meaning = Dense(10, name='dis_gen_meaning')(flatten_gen)
    dis_concat = Concatenate(name='dis_concat')([dis_gen_meaning, dis_meaning, dis_memory])
    dis_output = Dense(1, activation='sigmoid', name='dis_output')(dis_concat)

    # Setup the models.
    gen = Model(inputs=[gen_input, mem_input], outputs=gen_output, name='generator')
    dis = Model(inputs=[gen_input, dis_input, mem_input], outputs=dis_output, name='discriminator')

    # Build the full model. Use a copy of the discriminator so that it can be
    # trainable on its own but untrainable in the full model.
    full_output = clone_model(dis)([gen_input, gen_output, mem_input])
    full = Model(inputs=[gen_input, mem_input], outputs=full_output, name='full_model')

    return gen, dis, full
