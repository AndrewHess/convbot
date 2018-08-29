import argparse

def get_args():
    ''' Get all of the command line arguments. '''

    # Setup the parser.
    parser = argparse.ArgumentParser()

    # Add the arguments.
    parser.add_argument('--train', help='the parts of the model to train',
                        choices=['generator', 'discriminator', 'all', 'none'],
                        default='none')
    parser.add_argument('--load', type=str, help='the file to load the model from')
    parser.add_argument('--save', type=str, help='the file to save the model to')
    parser.add_argument('--model_folder', type=str, default='saved',
                        help='the directory that stores files for saving the model')

    return parser.parse_args()
