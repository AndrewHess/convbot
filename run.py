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


def format_input(encoded):
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

    return format_input(encoded)


def possibly_train_gen(gen, dis, full, data_x, data_y, args):
    ''' Train the generator if it is supposed to be trained. '''

    if args.train in ['all', 'gen']:
        print('traing the generator ...')
        full.fit(data_x, data_y)

        # Share the new weights with the discriminator and generator.
        share_weights(full, dis)
        share_weights(dis, full.get_layer('discriminator'))
        share_weights(full, gen)

    return


def possibly_train_dis(gen, dis, full, data_x, data_y, args):
    ''' Train the discriminator if it is supposed to be trained. '''

    if args.train in ['all', 'dis']:
        print('training the discriminator ...')
        dis.fit(data_x, data_y)

        # Share the new weights with the full model and the generator.
        share_weights(dis, full.get_layer('discriminator'))
        share_weights(dis, full)
        share_weights(dis, gen)

    return


def possibly_save(gen, dis, full, args):
    ''' Save the models if the user wants to. '''

    if args.save:
        print('enter s to save: ', end='')
        if input() != 's':
            return

        print('saving the model ...')
        save(full, args, prefix='full_')
        save(dis, args, prefix='dis_')
        save(gen, args, prefix='gen_')
        print('model saved')

    return


def talk(args, vocab, rev_vocab):
    ''' Infinitely run the loop of user and bot talking with user feedback. '''

    # Load the model.
    if args.load:
        gen, dis, full = load(args)

        # Share the weights.
        if args.share == 'gen':
            # Share the meaning and memory layers with the discriminator.
            share_weights(full, dis)
            share_weights(full, full.get_layer('discriminator'))
        else:
            # Share the meaning and memory layers with the untrainable
            # discriminator in the full model and the generator.
            share_weights(dis, full.get_layer('discriminator'))
            share_weights(dis, full)
    else:
        # Get the model.
        gen, dis, full = setup_model()

    # Run the main loop.
    while True:
        if args.train_file is not None:
            # Make sure at least one model is being trained.
            assert(args.train != 'none')

            train_x, train_y = [], []

            # Read the data from train_file.
            with open(os.path.join(args.data_folder, args.train_file), 'r') as infile:
                for line in infile:
                    line = line[:-1]  # Remove the newline.
                    pos = line.find(':')
                    train_x.append(line[:pos])
                    train_y.append(line[pos + 1:])

            # Set each item in train_x and train_y to what is used as input.
            for (i, (x, y)) in enumerate(zip(train_x, train_y)):
                # Encode the data into word id numbers.
                x = encode_with_dict(x.split(' '), vocab)
                y = encode_with_dict(y.split(' '), vocab)

                # Get the data into the input format for the models.
                x = format_input(x)
                y = format_input(y)

                train_x[i] = x
                train_y[i] = y

            # Get the generator predictions.
            pred = [gen.predict(x) for x in train_x]

            # Create the input for the discriminator.
            real_dis_input = np.concatenate([y[0] for y in train_y])
            prompt_input = np.concatenate([x[0] for x in train_x] * 2)
            word_input = np.concatenate((np.concatenate(pred), real_dis_input))
            mem_input = np.array([np.array([1])] * 2 * len(train_x))
            dis_input = [prompt_input, word_input, mem_input]

            # Create the input for the generator.
            gen_input = np.concatenate([x[0] for x in train_x])
            gen_input = [gen_input, np.array([[1]] * len(train_x))]

            # Create the labels.
            gen_labels = np.array([np.array([0])] * len(train_x))
            dis_labels = np.concatenate((np.array([np.array([0])] * len(train_x)),
                                         np.array([np.array([1])] * len(train_x))))

            # Train and save the models.
            possibly_train_gen(gen, dis, full, gen_input, gen_labels, args)
            possibly_train_dis(gen, dis, full, dis_input, dis_labels, args)
            possibly_save(gen, dis, full, args)
        else:
            print(prompt, end='')

            try:
                gen_input = get_formatted_user_input(vocab)
            except ValueError:
                continue

            response = gen.predict(gen_input)
            bad_gen_out = [np.array(response), np.array([[1]])]

            # Get the most likely word for each position.
            response = np.argmax(response[0], axis=1)

            # Print the response.
            print(prompt, ' '.join(encode_with_dict(response, rev_vocab)))

            # Train the model.
            if args.train != 'none':
                # Get the label response from the user.
                print('Enter a good response:', end=' ')
                try:
                    good_gen_out = get_formatted_user_input(vocab)
                except ValueError:
                    continue

                # Setup the input for training the discriminator.
                dis_input = [np.concatenate((gen_input[0], gen_input[0])),
                             np.concatenate((bad_gen_out[0], good_gen_out[0])),
                             np.concatenate((bad_gen_out[1], good_gen_out[1]))]

                # Train and save the models.
                possibly_train_gen(gen, dis, full, gen_input, np.array([0]), args)
                possibly_train_dis(gen, dis, full, dis_input, np.array([0, 1]), args)
                possibly_save(gen, dis, full, args)

    return


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
