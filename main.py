import numpy as np

from model import setup_model
from utils import losses


def main():
    # Get the model.
    gen, dis, full = setup_model()

    x = [np.array([[0] * 100]), np.array([[0]])]
    y = np.array([0])

    dis.fit(x, y)
    full.fit(x, y)

    return


if __name__ == '__main__':
    main()
