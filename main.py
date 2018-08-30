import os
import numpy as np

from model import setup_model
from run import add_data, talk
from utils import losses
from utils.args import get_args
from utils.preprocessing import make_vocab


def main():
    # Get the command line arguments.
    args = get_args()
    print('args:', args)

    # Get the model.
    gen, dis, full = setup_model()

    # Build the vocabulary.
    vocab, rev_vocab = make_vocab()

    if args.data_file is not None:
        # Get training data from the user.
        add_data(os.path.join(args.data_folder, args.data_file), vocab)
    else:
        # Run the bot.
        talk(args, gen, dis, full, vocab)

    return


if __name__ == '__main__':
    main()
