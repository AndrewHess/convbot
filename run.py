import os.path
import numpy as np
from keras.models import load_model

from utils.preprocessing import make_vocab, encode_with_dict
from utils.losses import generator_loss, discriminator_loss

prompt = '> '
num_words = 100
vocab_len = 5


def format_input(encoded, vocab):
    ''' Generate input for the model based on a list of numbers and the vocab. '''

    # The model input has a shape of (batch_size, num_words, vocab_len) and
    # a dummy input with a shape of (1,) that should have the value 1.

    # Zero pad the input.
    encoded += [0] * (100 - len(encoded))

    one_hot = np.zeros((num_words, vocab_len))
    one_hot[np.arange(100), encoded] = 1

    return [np.array([one_hot]), np.array([[1]])]


def get_formatted_user_input(vocab):
    ''' Get user input and format it for input into the model. '''
    user_input = input()

    try:
        encoded = encode_with_dict(user_input.split(' '), vocab)
    except ValueError:
        raise

    return format_input(encoded, vocab)


def run(args, gen, dis, full):
    ''' Infinitely run the loop of user and bot talking with user feedback. '''

    # Load the model.
    if args.load:
        full = load_model(os.path.join(args.model_folder, args.load),
                          custom_objects={'discriminator_loss': discriminator_loss,
                                          'generator_loss': generator_loss})

    # Build the vocabulary.
    vocab, rev_vocab = make_vocab()

    while True:
        print(prompt, end='')

        try:
            gen_input = get_formatted_user_input(vocab)
        except ValueError:
            continue

        response = gen.predict(gen_input)[0]

        # Get the most likely word for each position.
        response = np.argmax(response, axis=1)

        # Print the response.
        decoded = encode_with_dict(response, rev_vocab)
        print(prompt, ' '.join(decoded))

        # Train the model using the user to generate good labels.
        if args.train != 'none':
            # Get the label response from the user.
            print('Enter a good response:', end=' ')
            try:
                good_gen_out = get_formatted_user_input(vocab)
            except ValueError:
                continue

            # Setup the input for training the discriminator.
            bad_gen_out = format_input(list(response), vocab)
            dis_input = [np.concatenate((bad_gen_out[0], good_gen_out[0])),
                         np.concatenate((bad_gen_out[1], good_gen_out[1]))]

            # Train the model.
            if args.train in ['all', 'generator']:
                print('traing the generator ...')
                full.fit(gen_input, np.array([0]))

            # Train the discriminator.
            if args.train in ['all', 'discriminator']:
                print('training the discriminator ...')
                # dis.fit(dis_input, np.array([0, 1]))
                dis.fit(good_gen_out, np.array([1]))
                dis.fit(bad_gen_out,  np.array([0]))

            # Save the model.
            if args.save:
                print('saving model ...')
                path = os.path.join(args.model_folder, args.save)
                full.save(path)
                print('model saved')
