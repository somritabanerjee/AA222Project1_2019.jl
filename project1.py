import numpy as np


def optimize(f, g, x0, n, prob):
    """

    Arguments:
      - `f`: Function to be optimized
      - `g`: Gradient function for `f`
      - `x0`: (Vector) Initial position to start from
      - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
      - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
    """
    x_best = x0
    y_best = f(x0)
    for i in range(n - 1):
        x_next = x_best + np.random.randn(len(x_best))
        y_next = f(x_next)
        if y_next < y_best:
            x_best, y_best = x_next, y_next

    return x_best
