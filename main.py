import numpy as np

from model import setup_model
from run import run
from utils import losses


def main():
    # Get the model.
    gen, dis, full = setup_model()
    run(gen, dis, full)

    return


if __name__ == '__main__':
    main()
