using Random

"""

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
"""
function optimize(f, g, x0, n, prob)
    x_best = x0
    y_best = f(x0)
    for i in 1 : n-1
        x_next = x_best + randn(length(x_best))
        y_next = f(x_next)
        if y_next < y_best
            x_best, y_best = x_next, y_next
        end
    end
    return x_best
end
