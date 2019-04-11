using Random
using Plots
pyplot(size = (900,900), legend = true)


function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 5.0 * (x[2] - x[1]^2)^2
end

function g(x)
        return
end
"""

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
"""
function optimizeRandom(f, g, x0, n, prob)
    x_best = x0
    y_best = f(x0)
    for i in 1 : n-1
        x_next = x_best + randn(length(x_best))
        scatter!([x_next[1]],[x_next[2]],markercolor = :blue)
        y_next = f(x_next)
        if y_next < y_best
            x_best, y_best = x_next, y_next
        end
    end
    scatter!([x_best[1]],[x_best[2]],markercolor = :red, label="best found")
    return x_best
end

function optimize(f, g, x0, n, prob)
    a,b = fibonacci_search(f, [-3,-3], [3,3], n)
    x_best = (a+b)/2
    scatter!([x_best[1]],[x_best[2]],markercolor = :red, label="best found")
    return x_best
end

function fibonacci_search(f, a, b, n; ε=0.01)
    φ = (1+√5)/2
    s = (1-√5)/(1+√5)
    ρ = 1 / (φ*(1-s^(n+1))/(1-s^n))
    d = ρ*b + (1 - ρ)*a
    yd = f(d)
    scatter!([d[1]],[d[2]],markercolor = :blue,label="")
    for i in 1 : n - 1
        if i == n - 1
            c = ε*a + (1-ε)*d
        else
            c = ρ*a + (1 - ρ)*b
        end
        yc = f(c)
        scatter!([c[1]],[c[2]],markercolor = :blue,label="")
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
        ρ = 1 / (φ*(1-s^(n-i+1))/(1-s^(n-i)))
    end
    return a < b ? (a, b) : (b, a)
end

X = range(-5, length=100, stop=5)
Y = range(-5, length=100, stop=5)
Z = zeros(length(X),length(Y))
for i in 1:length(X)
    for j in 1:length(Y)
        Z[i,j]=rosenbrock([X[j],Y[i]])
    end
end
contour(X,Y,Z, levels= 50)
scatter!([1],[1],markercolor=:green,markersize=7,label="true minimum")
optimize(rosenbrock,g,[0.5,1],10,0)
gui()
