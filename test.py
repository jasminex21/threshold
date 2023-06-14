import numpy as np

def square(x):
    """Returns the squared elements in array x.

    Args:
        x: an ndarray whose elements will be squared.
    """

    return x**2


if __name__ == '__main__':

    a = np.random.random((3, 10))

    result = square(a)

