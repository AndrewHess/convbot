import argparse

def get_args():
    ''' Get all of the command line arguments. '''

    # Setup the parser.
    parser = argparse.ArgumentParser()

    # Setup general training flags.
    parser.add_argument('--train', help='the parts of the model to train',
                        choices=['gen', 'dis', 'all', 'none'], default='none')
    parser.add_argument('--save_itr', type=int, default=1,
                        help='the number of iterations to train before asking to save')
    parser.add_argument('--min_gen_loss', type=float, default=0.5,
                        help='the generator threshold loss to switch to training the discriminator')
    parser.add_argument('--min_dis_loss', type=float, default=0.5,
                        help='the discriminator threshold loss to switch to training the generator')
    parser.add_argument('--setup_meaning', default=False, action='store_true',
                        help='train the meaning model')

    # Setup flags related to saving and loading.
    parser.add_argument('--model_folder', type=str, default='./saved',
                        help='the directory that stores files for saving the model')
    parser.add_argument('--load', type=str, help='the file to load the model from')
    parser.add_argument('--save', type=str, help='the file to save the model to')
    parser.add_argument('--load_meaning', type=str,
                        help='the file to load the meaning model from')
    parser.add_argument('--save_meaning', type=str, default='meaning_default',
                        help='the file to save the meaning model to')

    # Setup flags related to training from a file.
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='the folder to store training data files in')
    parser.add_argument('--data_file', type=str, default=None,
                        help='the file to add data to for the model to learn from')
    parser.add_argument('--train_file', type=str, default=None,
                        help='the file to use as training data')

    return parser.parse_args()
