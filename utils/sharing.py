import os
from keras.models import load_model

from utils.losses import generator_loss, discriminator_loss


def load(gen, dis, full, args):
    ''' Load the GAN models. '''
    assert(args.load is not None)

    custom_objects = {'discriminator_loss': discriminator_loss,
                      'generator_loss': generator_loss}
    full = load_model(os.path.join(args.model_folder, 'full_' + args.load),
                      custom_objects=custom_objects)
    dis  = load_model(os.path.join(args.model_folder, 'dis_' + args.load),
                      custom_objects=custom_objects)

    return gen, dis, full


def save(model, args, prefix=''):
    ''' Save a model. '''
    assert(args.save is not None)
    model.save(os.path.join(args.model_folder, prefix + args.save))

    return


def share_weights(src_model, dst_model):
    ''' Set weights in one model to the weights in another by layer name. '''
    tempfile = '.temp_weights.h5'

    src_model.save_weights(tempfile)
    dst_model.load_weights(tempfile, by_name=True)
    os.remove(tempfile)

    return
