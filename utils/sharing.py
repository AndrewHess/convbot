import os
from keras.models import load_model

from utils.losses import generator_loss, discriminator_loss


def load(args):
    ''' Load the GAN models. '''
    assert(args.load is not None)

    custom_objects = {'discriminator_loss': discriminator_loss,
                      'generator_loss': generator_loss}
    full = load_model(os.path.join(args.model_folder, 'full_' + args.load),
                      custom_objects=custom_objects)
    dis  = load_model(os.path.join(args.model_folder, 'dis_' + args.load),
                      custom_objects=custom_objects)
    gen  = load_model(os.path.join(args.model_folder, 'gen_' + args.load),
                      custom_objects=custom_objects)

    return gen, dis, full


def save(model, args, prefix=''):
    ''' Save a model. '''
    assert(args.save is not None)
    model.save(os.path.join(args.model_folder, prefix + args.save))

    return


def share_weights(src_model, dst_model):
    ''' Set weights in one model to the weights in another by layer name. '''

    for s_layer in src_model.layers:
        for d_layer in dst_model.layers:
            # Share exact layers.
            if s_layer.name == d_layer.name:
                d_layer.set_weights(s_layer.get_weights())

            # Don't share input and output layers unless they are exact matches.
            if 'input' in s_layer.name or 'output' in s_layer.name:
                continue

            # Share the layer weights if the layers represent the same thing.
            # Layer names are prefixed with gen_ or dis_, so look after the
            # first 3 characters.
            if s_layer.name[3:] == d_layer.name[3:]:
                d_layer.set_weights(s_layer.get_weights())

    return
