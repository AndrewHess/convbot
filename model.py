from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from utils.losses import generator_loss, discriminator_loss


input_size = 100


def setup_model():
    # Build the models.
    gen, dis, full = build_model()

    # Freeze the discriminator layers of the full model.
    for layer in full.layers:
        if 'dis' in layer.name:
            layer.trainable = False

    full.get_layer('discriminator').trainable = False

    print('full model')
    print('--------------------------------')
    for layer in full.layers:
        print(layer.name, layer.trainable)
    print('--------------------------------')


    print('dis model')
    print('--------------------------------')
    for layer in dis.layers:
        print(layer.name, layer.trainable)
    print('--------------------------------')


    # Compile the models for training.
    dis.compile(optimizer='adam', loss=discriminator_loss)
    full.compile(optimizer='adam', loss=generator_loss)

    return gen, dis, full


def build_model():
    meaning_model = build_meaning()
    mem_input = Input(shape=(1,), name='mem_input')
    memory = Dense(2, name='memory')(mem_input)

    # Build the generator.
    gen_input = Input(shape=(input_size,), name='gen_input')
    meaning = meaning_model(gen_input)
    concat = Concatenate(name='gen_concat')([memory, meaning])
    gen_output = Dense(input_size, name='gen_output')(concat)

    # Build the discriminator.
    dis_input = Input(shape=(input_size,), name='dis_input')
    meaning = meaning_model(dis_input)
    concat = Concatenate(name='dis_concat')([memory, meaning])
    dis_output = Dense(1, activation='sigmoid', name='dis_output')(concat)

    # Setup the models.
    gen  = Model(inputs=[gen_input, mem_input], outputs=gen_output, name='generator')
    dis  = Model(inputs=[dis_input, mem_input], outputs=dis_output, name='discriminator')

    full_output = dis([gen_output, mem_input])
    full = Model(inputs=[gen_input, mem_input], outputs=full_output, name='full_model')

    return gen, dis, full


def build_meaning():
    ''' Build a network that determines the meaning of a sentence. '''
    input_layer = Input(shape=(input_size,), name='meaning_input')
    output_layer = Dense(2, activation='relu')(input_layer)


    return Model(inputs=input_layer, outputs=output_layer, name='meaning')
