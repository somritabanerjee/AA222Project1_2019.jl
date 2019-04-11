using Random
# using Plots
# pyplot(size = (900,900), legend = true)


# function rosenbrock(x::Vector)
#     return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# end

# function rosenbrock_gradient(x::Vector)
#     storage = zeros(2)
#     storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
#     storage[2] = 200.0 * (x[2] - x[1]^2)
#     return storage
# end
"""

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
"""
function optimize(f, g, x0, n, prob)
    if prob in ["simple_1","simple_2","simple_3"]
        optimizeHookeJeeves(f, g, x0, n, prob)
    elseif prob in ["secret_1"]
        optimizeNesterov(f, g, x0, n, prob, 0.001, 0.9)
    elseif prob in ["secret_2"]
        optimizeAdagrad(f, g, x0, n, prob, 0.01, 1.0e-8)
    end
end

function optimizeAdagrad(f, g, x0, n, prob, α, ε)
    M = Adagrad(α, ε, zeros(length(x0)))
    M = initAda!(M, f, g, x0)
    evalsLeft = n
    x_best = x0
    while evalsLeft >=2
        x_best = stepAda(M, f, g, x_best)
        evalsLeft = evalsLeft - 2
        # print(x_best)
    end
    # scatter!([x_best[1]],[x_best[2]],markercolor = :red, label="best found")
    return x_best
end

abstract type DescentMethod end

mutable struct Adagrad <: DescentMethod
    α # learning rate
    ε # small value
    s # sum of square gradient
end
function initAda!(M::Adagrad, f, ∇f, x)
    M.s = zeros(length(x))
    return M
end
function stepAda(M::Adagrad, f, ∇f, x)
    α, ε, s, g = M.α, M.ε, M.s, ∇f(x)
    s[:] += g.*g
    return x - α*g ./ (sqrt.(s) .+ ε)
end

function optimizeNesterov(f, g, x0, n, prob, α, β)
    M = NesterovMomentum(α, β, zeros(length(x0)))
    M = initNesterov!(M, f, g, x0)
    evalsLeft = n
    x_best = x0
    while evalsLeft >=2
        x_best = stepNesterov(M, f, g, x_best)
        evalsLeft = evalsLeft - 2
        # print(x_best)
    end
    # scatter!([x_best[1]],[x_best[2]],markercolor = :red, label="best found")
    return x_best
end

mutable struct NesterovMomentum <: DescentMethod
    α # learning rate
    β # momentum decay
    v # momentum
end
function initNesterov!(M::NesterovMomentum, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function stepNesterov(M::NesterovMomentum, f, ∇f, x)
    α, β, v = M.α, M.β, M.v
    v[:] = β*v - α*∇f(x + β*v)
    return x + v
end


function optimizeHookeJeeves(f, g, x0, n, prob)
    α = 1;
    x_best = hooke_jeeves(f, x0, α, n)
    # scatter!([x_best[1]],[x_best[2]],markercolor = :red, label="best found")
    return x_best
end

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]

function hooke_jeeves(f, x, α, maxEval, γ=0.5)
    y, n = f(x), length(x)
    # scatter!([x[1]],[x[2]],markercolor = :blue,label="")
    evalsLeft = maxEval - 1
    while evalsLeft > 0
        improved = false
        for i in 1 : n
            x′ = x + α*basis(i, n)
            if evalsLeft <= 0
                return x 
            end
            y′ = f(x′)
            # scatter!([x′[1]],[x′[2]],markercolor = :blue,label="")
            evalsLeft = evalsLeft - 1
            if y′ < y
                x, y, improved = x′, y′, true
            else
                x′ = x - α*basis(i, n)
                if evalsLeft <= 0
                    return x 
                end
                y′ = f(x′)
                # scatter!([x′[1]],[x′[2]],markercolor = :blue,label="")
                evalsLeft = evalsLeft - 1
                if y′ < y
                    x, y, improved = x′, y′, true
                end
            end
        end
        if !improved
            α *= γ
        end
    end
    return x
end

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

function optimizeFibonacci(f, g, x0, n, prob)
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

# X = range(-5, length=100, stop=5)
# Y = range(-5, length=100, stop=5)
# Z = zeros(length(X),length(Y))
# for i in 1:length(X)
#     for j in 1:length(Y)
#         Z[i,j]=rosenbrock([X[j],Y[i]])
#     end
# end
# contour(X,Y,Z, levels= 50)
# scatter!([1],[1],markercolor=:green,markersize=7,label="true minimum")
# optimize(rosenbrock,rosenbrock_gradient,[2,2],20,"simple_1")
# gui()
