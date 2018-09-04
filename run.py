import os
import numpy as np
from keras.models import load_model

from model import setup_model
from utils.preprocessing import encode_with_dict, text_in_vocab
from utils.losses import generator_loss, discriminator_loss
from utils.sharing import load, save, share_weights

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


def talk(args, vocab, rev_vocab):
    ''' Infinitely run the loop of user and bot talking with user feedback. '''

    # Load the model.
    if args.load:
        gen, dis, full = load(args)
        
        # Share the weights.
        if args.share == 'gen':
            # Share the meaning and memory layers.
            share_weights(full, dis)

            # Share the new meaning and memory layers with the untrainable
            # discriminator in the full model.
            share_weights(dis, full.get_layer('discriminator'))
        else:
            # Share the meaning and memory layers with the untrainable
            # discriminator in the full model.
            # print('old full dis mem:', full.get_layer('discriminator').get_layer('memory').get_weights())
            share_weights(dis, full.get_layer('discriminator'))
            # print('new full dis mem:', full.get_layer('discriminator').get_layer('memory').get_weights())
            # print('dis mem:', dis.get_layer('memory').get_weights())

            # Share the meaning and memory layers with the generator.
            share_weights(dis, full)
    else:
        # Get the model.
        gen, dis, full = setup_model()

    # Run the main loop.
    while True:
        print(prompt, end='')

        try:
            gen_input = get_formatted_user_input(vocab)
        except ValueError:
            continue

        response = gen.predict(gen_input)
        bad_gen_out = [np.array(response), np.array([[1]])]
        # print('response:', response)
        # print('bad_gen_out:', bad_gen_out)

        # Get the most likely word for each position.
        response = np.argmax(response[0], axis=1)

        # Print the response.
        # decoded = encode_with_dict(response, rev_vocab)
        print(prompt, ' '.join(encode_with_dict(response, rev_vocab)))

        # Train the model using the user to generate good labels.
        if args.train != 'none':
            # Get the label response from the user.
            print('Enter a good response:', end=' ')
            try:
                good_gen_out = get_formatted_user_input(vocab)
            except ValueError:
                continue

            # print('good_gen_out:', good_gen_out)

            # Setup the input for training the discriminator.
            # dis_input = [np.concatenate((bad_gen_out[0], good_gen_out[0])),
            #              np.concatenate((bad_gen_out[1], good_gen_out[1]))]

            # save(dis, args)

            # Train the model.
            if args.train in ['all', 'gen']:
                print('traing the generator ...')
                # print('old full mem:', full.get_layer('memory').get_weights())
                # print('old full dis mem:', full.get_layer('discriminator').get_layer('memory').get_weights())
                full.fit(gen_input, np.array([0]))

                # Share the new weights with the discriminator.
                share_weights(full, dis)
                share_weights(full, full.get_layer('discriminator'))

                # Share the new weights with the generator.
                share_weights(full, gen)

                # print('new full mem:', full.get_layer('memory').get_weights())
                # print('new full dis mem:', full.get_layer('discriminator').get_layer('memory').get_weights())

            # dis  = load_model(os.path.join(args.model_folder, 'temp.h5'),
            #                   custom_objects={'discriminator_loss': discriminator_loss})

            # Train the discriminator.
            if args.train in ['all', 'dis']:
                print('training the discriminator ...')
                # print('old dis mem:', dis.get_layer('memory').get_weights())
                # dis.fit(dis_input, np.array([0, 1]))
                dis.fit(good_gen_out, np.array([1]))
                dis.fit(bad_gen_out,  np.array([0]))
                # print('new dis mem:', dis.get_layer('memory').get_weights())

                # Share the new weights with the full model.
                share_weights(dis, full.get_layer('discriminator'))
                share_weights(dis, full)

                # Share the weights with the generator.
                share_weights(dis, gen)

            # Save the model.
            if args.save:
                print('enter s to save: ', end='')
                if input() != 's':
                    continue

                print('saving the model ...')
                save(full, args, prefix='full_')
                save(dis, args, prefix='dis_')
                save(gen, args, prefix='gen_')
                print('model saved')


def add_data(filename, vocab):
    ''' Infinitely get a user prompt and then a good response from the user. '''

    # Write to the end of the file and create the file if necessary.
    while True:
        # Get the input from the user.
        print(prompt, end='')
        user_input = input()

        if not text_in_vocab(user_input, vocab):
            print('Found a word not in the vocab')
            continue

        # Get a good response from the user.
        print('Enter a good response: ', end='')
        user_response = input()

        if not text_in_vocab(user_response, vocab):
            print('Found a word not in the vocab')
            continue

        # Add the input and response to the data file.
        with open(filename, 'a+') as outfile:
            outfile.write(user_input + ':' + user_response + '\n')
