import os
import numpy as np

from run import add_data, talk, train_meaning
from utils import losses
from utils.args import get_args
from utils.preprocessing import make_vocab


def main():
    # Get the command line arguments.
    args = get_args()

    # Build the vocabulary.
    vocab, rev_vocab = make_vocab()

    if args.setup_meaning:
        # Train the meaning autoencoder.
        train_meaning(args)
    elif args.data_file is not None:
        # Get training data from the user.
        add_data(os.path.join(args.data_folder, args.data_file), vocab)
    else:
        # Run the bot.
        talk(args, vocab, rev_vocab)

    return


if __name__ == '__main__':
    main()
