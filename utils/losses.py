import keras.backend as K
import tensorflow as tf


def generator_loss(true, pred):
    '''
    true: A series of boolean values for whether the input was produced by the
          human or by the generator.
    pred: A series of probabilies for whether the input was produced by the
          human or by the generator.
    '''

    # The loss in the original GAN paper is mean(log(1 - D(G(z)))). However,
    # this gives NaNs in the loss for some reason, so we use a different loss.
    # return K.mean(K.log(K.ones(shape=tf.shape(pred)) - pred))

    return K.mean(1 - K.abs(true - pred))


def discriminator_loss(true, pred):
    '''
    true: A series of boolean values for whether the input was produced by the
          human or by the generator.
    pred: A series of probabilies for whether the input was produced by the
          human or by the generator.
    '''

    # The loss in the original GAN paper is mean(log(D(x)) + log(1 - D(G(z)))).
    # However, this gives NaNs in the loss, so we use a different loss.
    # return K.mean(K.log(K.abs(true - pred)))

    return K.mean(K.abs(true - pred))
