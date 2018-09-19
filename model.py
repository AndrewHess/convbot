import os
import tensorflow as tf
from tensorflow import shape
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Lambda, Reshape, Flatten
from keras.optimizers import adam

from utils.losses import generator_loss, discriminator_loss


input_size = 10
vocab_len = 20


def setup_model(args):
    ''' Build and compile the models. '''

    # Build the models.
    gen, dis, full = build_model()
    meaning = build_meaining_autoencoder()

    meaning = None
    if args.load_meaning:
        meaning = load_model(os.path.join(args.model_folder, args.load_meaning))
    else:
        print('warning: using untrained parameters in the meaning model')
        meaning = build_meaining_autoencoder()

    # Use the meaing weights found by the meaning model.
    encoding = meaning.get_layer('meaning_ae_hidden').get_weights()
    decoding = meaning.get_layer('meaning_ae_output').get_weights()

    dis.get_layer('dis_meaning').set_weights(encoding)
    dis.get_layer('dis_gen_meaning').set_weights(encoding)
    full.get_layer('discriminator').get_layer('dis_meaning').set_weights(encoding)
    full.get_layer('discriminator').get_layer('dis_gen_meaning').set_weights(encoding)

    gen.get_layer('gen_meaning').set_weights(encoding)
    gen.get_layer('gen_response').set_weights(decoding)
    full.get_layer('gen_meaning').set_weights(encoding)
    full.get_layer('gen_response').set_weights(decoding)

    # Compile the models for training.
    dis.compile(optimizer='adam', loss=discriminator_loss)

    # The full model is for training the generator so freeze its discriminator.
    full.get_layer('discriminator').trainable = False

    full.compile(optimizer='adam', loss=generator_loss)

    meaning.summary()
    gen.summary()
    dis.summary()
    full.summary()

    return gen, dis, full


def setup_meaning_autoencoder():
    ''' Build and compile the autoencoder model. '''

    auto = build_meaining_autoencoder()
    auto.compile(optimizer='adam', loss='mean_absolute_error')

    return auto


def build_model():
    ''' Build the generator, discriminator, and combined models. '''

    # Setup the memory.
    mem_input = Input(shape=(1,), name='mem_input')
    gen_memory = Dense(2, name='gen_memory')(mem_input)
    dis_memory = Dense(2, name='dis_memory')(mem_input)

    # Build the generator.
    gen_input = Input(shape=(input_size, vocab_len), name='gen_input')
    flatten = Flatten()(gen_input)
    gen_meaning = Dense(10, trainable=False, name='gen_meaning')(flatten)
    gen_mem_concat = Concatenate(name='gen_mem_concat')([gen_meaning, gen_memory])

    hidden = Dense(10, name='gen_meaning_to_response')(gen_mem_concat)

    # Use the autoencoder to decode the response.
    response = Dense(input_size * vocab_len, name='gen_response')(hidden)
    gen_output = Reshape((input_size, vocab_len), name='gen_output')(response)

    # Build the discriminator.
    dis_input = Input(shape=(input_size, vocab_len), name='dis_input')
    flatten = Flatten()(dis_input)
    dis_meaning = Dense(10, trainable=False, name='dis_meaning')(flatten)
    flatten = Flatten()(gen_input)
    dis_gen_meaning = Dense(10, trainable=False, name='dis_gen_meaning')(flatten)
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


def build_meaining_autoencoder():
    ''' Build an autoencoder model to determine meaning of a sentence. '''

    input_layer = Input(shape=(input_size * vocab_len,), name='meaning_ae_input')
    hidden = Dense(10, name='meaning_ae_hidden')(input_layer)
    output_layer = Dense(input_size * vocab_len, name='meaning_ae_output')(hidden)

    return Model(inputs=input_layer, outputs=output_layer, name='meaning_ae')
