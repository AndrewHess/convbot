import numpy as np

from model import setup_model
from run import run
from utils import losses
from utils.args import get_args


def main():
    # Get the command line arguments.
    args = get_args()

    print('args:', args)

    # Get the model.
    gen, dis, full = setup_model()

    # Run the convbot.
    run(args, gen, dis, full)

    return


if __name__ == '__main__':
    main()
