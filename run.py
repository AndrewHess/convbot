import numpy as np

from utils.preprocessing import make_vocab, encode_with_dict

prompt = '> '


def get_formatted_user_input(vocab):
    ''' Get user input and format it for input into the model. '''
    user_input = input()

    try:
        encoded = encode_with_dict(user_input.split(' '), vocab)
    except ValueError:
        raise

    # Feed the input to the generator. It expects an input of shape (100,)
    # and a dummy input of shape (1,) that should have the value 1.
    encoded += [0] * (100 - len(encoded))  # Zero pad the input.
    gen_input = [np.array([encoded]), np.array([[1]])]

    return gen_input


def run(gen, dis, full, train=True):
    ''' Infinitely run the loop of user and bot talking with user feedback. '''

    # Build the vocabulary.
    vocab, rev_vocab = make_vocab()

    while True:
        print(prompt, end='')

        try:
            gen_input = get_formatted_user_input(vocab)
        except ValueError:
            continue

        response = gen.predict(gen_input)[0]
        print('raw output:', response)

        # Round the response to integers.
        response = [int(round(num)) for num in response]

        # Print the response.
        decoded = encode_with_dict(response, rev_vocab)
        print('response:', ' '.join(decoded))

        # Train the model using the user to generate good labels.
        if train:
            # Get the label response from the user.
            print('Enter a good response:', end=' ')
            try:
                good_gen_out = get_formatted_user_input(vocab)
            except ValueError:
                continue

            # Setup the input for training the discriminator.
            bad_gen_out = [np.array([response]), np.array([[1]])]
            dis_input = [np.concatenate((bad_gen_out[0], good_gen_out[0])),
                         np.concatenate((bad_gen_out[1], good_gen_out[1]))]
                         
            # Train the model.
            full.fit(gen_input, np.array([1]))
            dis.fit(dis_input, np.array([0, 1]))
