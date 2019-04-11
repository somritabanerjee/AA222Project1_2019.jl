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
    a,b = fibonacci_search(f, [-3,-3], [3,3], n)
    x_best = (a+b)/2
    return x_best
end

function fibonacci_search(f, a, b, n; ε=0.01)
    φ = (1+√5)/2
    s = (1-√5)/(1+√5)
    ρ = 1 / (φ*(1-s^(n+1))/(1-s^n))
    d = ρ*b + (1 - ρ)*a
    yd = f(d)
    for i in 1 : n - 1
        if i == n - 1
            c = ε*a + (1-ε)*d
        else
            c = ρ*a + (1 - ρ)*b
        end
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
        ρ = 1 / (φ*(1-s^(n-i+1))/(1-s^(n-i)))
    end
    return a < b ? (a, b) : (b, a)
end