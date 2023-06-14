import numpy as np

def square(x):
    """Returns the squared elements in x."""

    return x**2


if __name__ == '__main__':

    a = np.random.random((3, 10))

    result = square(a)

