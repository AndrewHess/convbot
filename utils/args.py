import argparse

def get_args():
    ''' Get all of the command line arguments. '''

    # Setup the parser.
    parser = argparse.ArgumentParser()

    # Add the arguments.
    parser.add_argument('--train', help='the parts of the model to train',
                        choices=['gen', 'dis', 'all', 'none'], default='none')
    parser.add_argument('--load', type=str, help='the file to load the model from')
    parser.add_argument('--save', type=str, help='the file to save the model to')
    parser.add_argument('--model_folder', type=str, default='./saved',
                        help='the directory that stores files for saving the model')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='the folder to store training data files in')
    parser.add_argument('--data_file', type=str, default=None,
                        help='the file to add data to for the model to learn from')
    parser.add_argument('--train_file', type=str, default=None,
                        help='the file to use as training data')
    parser.add_argument('--save_itr', type=int, default=1,
                        help='the number of iterations to train before asking to save')

    return parser.parse_args()
