import tensorflow.keras.backend as K


def generator_loss(true, pred):
    '''
    true: A series of boolean values for whether the input was produced by the
          human or by the generator.
    pred: A series of probabilies for whether the input was produced by the
          human or by the generator.
    '''

    # The values in pred must be between 0 and 1.
    return K.mean(1 - K.abs(true - pred))


def discriminator_loss(true, pred):
    '''
    true: A series of boolean values for whether the input was produced by the
          human or by the generator.
    pred: A series of probabilies for whether the input was produced by the
          human or by the generator.
    '''

    # The values in pred must be between 0 and 1.
    return K.mean(K.abs(true - pred))
