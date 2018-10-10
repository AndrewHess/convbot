import os
import numpy as np
from keras.models import load_model

from model import setup_model
from utils.preprocessing import encode_with_dict, text_in_vocab
from utils.losses import generator_loss, discriminator_loss
from utils.sharing import load, save, share_weights

prompt = '> '
num_words = 10
num_rand = 100
vocab_len = 20
itr = 0


def format_input(encoded):
    ''' Generate input for the model based on a list of numbers and the vocab. '''

    # The model input has a shape of (batch_size, num_words, vocab_len) and
    # an input of shape (batch_size, 10).

    # Zero pad the input.
    encoded += [0] * (num_words - len(encoded))

    one_hot = np.zeros((num_words, vocab_len))
    one_hot[np.arange(num_words), encoded] = 1

    return [np.array([one_hot]), np.array([np.random.random_sample(size=(num_rand,))])]


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
        hist = full.fit(data_x, data_y)

        # Check if the model to train should be toggled.
        if hist.history['loss'][0] < args.min_gen_loss:
            args.train = 'dis'
        elif hist.history['loss'][0] < args.max_gen_loss:
            args.train = 'all'

        # Share the new weights with the generator.
        share_weights(full, gen)

        # The full model should have the weights of the generator and the
        # discriminator models.
        for w1, w2 in zip(dis.get_weights(), full.get_layer('discriminator').get_weights()):
            assert(not np.any(w1 - w2))

        for w1, w2 in zip(gen.get_weights(), full.get_weights()):
            assert(not np.any(w1 - w2))

    return


def possibly_train_dis(gen, dis, full, data_x, data_y, args):
    ''' Train the discriminator if it is supposed to be trained.

        The discriminator is trained on batches of only real output and only
        generated output as suggested on https://github.com/soumith/ganhacks.
    '''

    if args.train in ['all', 'dis']:
        # Get the real batch and the generated batch using the noisy data_y.
        real_data_x, fake_data_x = [[], []], [[], []]
        real_data_y, fake_data_y = [], []

        for i, item in enumerate(data_y):
            # Check if the data is real or generated.
            if item[0] - 0.5 > 0:
                fake_data_y.append(item)
                # fake_data_x.append([data_x[0][i], data_x[1][i]])
                fake_data_x[0].append(data_x[0][i])
                fake_data_x[1].append(data_x[1][i])
            else:
                real_data_y.append(item)
                real_data_x[0].append(data_x[0][i])
                real_data_x[1].append(data_x[1][i])
                # real_data_x.append([data_x[0][i], data_x[1][i]])

        # Convert to numpy arrays.
        fake_data_y = np.array(fake_data_y)
        real_data_y = np.array(real_data_y)
        fake_data_x[0] = np.array(fake_data_x[0])
        fake_data_x[1] = np.array(fake_data_x[1])
        real_data_x[0] = np.array(real_data_x[0])
        real_data_x[1] = np.array(real_data_x[1])

        print('training the discriminator ...')
        hist_real_data = dis.fit(real_data_x, real_data_y)
        hist_fake_data = dis.fit(fake_data_x, fake_data_y)

        # Get the average loss.
        loss = (hist_real_data.history['loss'][0] + hist_fake_data.history['loss'][0]) / 2

        # Check if the model to train should be toggled.
        if loss < args.min_dis_loss:
            args.train = 'gen'
        elif loss < args.max_dis_loss:
            args.train = 'all'

        # Share the new weights with the full model.
        share_weights(dis, full.get_layer('discriminator'))

        # The full model should have the updated discriminator weights.
        for w1, w2 in zip(dis.get_weights(), full.get_layer('discriminator').get_weights()):
            assert(not np.any(w1 - w2))

    return


def possibly_save(gen, dis, full, args):
    ''' Save the models if the user wants to. '''

    global itr
    itr += 1
    print('iteration:', itr)

    if args.save and itr % args.save_itr == 0:
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

    # Setup the models.
    gen, dis, full = load(args) if args.load else setup_model(args)

    # Setup the training data if it is from a file.
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

    # Run the main loop.
    while True:
        if args.train_file is not None:
            # Use new random numbers for the input.
            train_x = [[x[0], np.random.random_sample(size=(1, num_rand))] for x in train_x]

            # Get the generator predictions.
            pred = [gen.predict(x) for x in train_x]

            # Create the input for the discriminator.
            real_dis_input = np.concatenate([y[0] for y in train_y])
            prompt_input = np.concatenate([x[0] for x in train_x] * 2)
            word_input = np.concatenate((np.concatenate(pred), real_dis_input))
            dis_input = [prompt_input, word_input]

            # Create the input for the generator.
            gen_input = np.concatenate([x[0] for x in train_x])
            gen_input = [gen_input, np.random.random_sample(size=(len(train_x), num_rand))]

            # Create the noisy labels.
            gen_labels = 1 - np.random.random_sample(size=(len(train_x), 1)) / 10
            dis_labels = np.concatenate((1 - np.random.random_sample(size=(len(train_x), 1)) / 10,
                                         np.random.random_sample(size=(len(train_x), 1)) / 10))

            # Randomly flip 5 percent of the discriminator's labels to keep the
            # discriminator loss from decreasing to 0 too quickly.
            for _ in range(int(len(train_x) * 0.05)):
                i = np.random.randint(0, len(train_x))
                k = len(train_x) - i - 1
                dis_labels[i][0], dis_labels[k][0] = dis_labels[k][0], dis_labels[i][0]

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
                             np.concatenate((bad_gen_out[0], good_gen_out[0]))]

                # Use noisy labels to help with training.
                gen_labels = np.random.random_sample(size=(1, 1)) / 10
                dis_labels = np.random.random_sample(size=(2, 1)) / 10
                gen_labels[0][0] = 1 - gen_labels[0][0]
                dis_labels[0][0] = 1 - dis_labels[0][0]

                # Train and save the models.
                possibly_train_gen(gen, dis, full, gen_input, gen_labels, args)
                possibly_train_dis(gen, dis, full, dis_input, dis_labels, args)
                possibly_save(gen, dis, full, args)

    # It will never reach this point.
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
