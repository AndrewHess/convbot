vocab_path = './vocab.txt'

def make_vocab():
    ''' Create the vocab dictionary. '''

    # Read the vocab file.
    vocab, rev_vocab = {}, {}
    counter = 0

    with open(vocab_path) as infile:
        for line in infile:
            # Don't include the newline.
            line = line[:-1]

            # Add the entries to the dictionaries.
            vocab[line] = counter
            rev_vocab[counter] = line

            counter += 1

    return vocab, rev_vocab


def text_in_vocab(text, vocab):
    ''' Check if each word in the text string is in the vocab. '''

    for word in text.split(' '):
        if word not in vocab:
            return False

    return True


def encode_with_dict(inputs, vocab):
    ''' Encode the input by replacing each value with its value from vocab. '''

    encoded = []

    for item in inputs:
        # Make sure the word is valid.
        if item not in vocab:
            print(f'ERROR: \'{item}\' is not in the vocabulary')
            print('vocab:', vocab)
            print('item type:', type(item))

            raise ValueError

        # Add the encoded word.
        encoded.append(vocab[item])

    return encoded
