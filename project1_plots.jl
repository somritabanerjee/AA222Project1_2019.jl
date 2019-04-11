using Random
using Plots
pyplot(size = (600,1100), legend = true)

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
    else
        print("Error: Only use this function for the simple problems")    
    end
end

function optimizeHookeJeeves(f, g, x0, n, prob; plotType = 1)
    α = 1;
    x_best = hooke_jeeves(f, x0, α, n, 0.5, plotType = plotType)
    if plotType ==1
        scatter!([x_best[1]],[x_best[2]],markercolor = :red, markersize=6, label="best found")
    end
    return x_best
end

basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1 : n]

function hooke_jeeves(f, x, α, maxEval, γ=0.5; plotType = 1)
    y, n = f(x), length(x)
    evalsLeft = maxEval - 1
    if plotType == 1
        scatter!([x[1]],[x[2]],markercolor = :blue,label="")
    elseif plotType == 2
        scatter!([maxEval - evalsLeft],[y],markercolor = :blue,label="")
    end
    
    while evalsLeft > 0
        improved = false
        for i in 1 : n
            x′ = x + α*basis(i, n)
            if evalsLeft <= 0
                return x 
            end
            y′ = f(x′)
            evalsLeft = evalsLeft - 1
            if plotType ==1
                scatter!([x′[1]],[x′[2]],markercolor = :blue,label="")
            elseif plotType == 2
                scatter!([maxEval - evalsLeft],[y′],markercolor = :blue,label="")
            end            
            if y′ < y
                x, y, improved = x′, y′, true
            else
                x′ = x - α*basis(i, n)
                if evalsLeft <= 0
                    return x 
                end
                y′ = f(x′)
                evalsLeft = evalsLeft - 1
                if plotType ==1
                    scatter!([x′[1]],[x′[2]],markercolor = :blue,label="")
                elseif plotType == 2
                    scatter!([maxEval - evalsLeft],[y′],markercolor = :blue,label="")
                end                
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




# Rosenbrock's
function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    return storage
end

# Powell's
function powell(x::Vector)
    return (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
        (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
end

function powell_gradient(x::Vector)
    storage = zeros(4)
    storage[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
    storage[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
    storage[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
    storage[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
    return storage
end

# Himmelblau's
function himmelblau(x::Vector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function himmelblau_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    return storage
end

function create_rosenbrock_plt(x0)
    X = range(-5, length=100, stop=5)
    Y = range(-5, length=100, stop=5)
    Z = zeros(length(X),length(Y))
    for i in 1:length(X)
        for j in 1:length(Y)
            Z[i,j]=rosenbrock([X[j],Y[i]])
        end
    end
    plt = contour(X,Y,Z, levels= 50, label = "", contour_labels= false, contours = false)
    scatter!([1],[1],markercolor=:green,markersize=8,label="true minimum")
    optimize(rosenbrock,rosenbrock_gradient,x0,50,"simple_1")
    title!("Rosenbrock function convergence with x0=$x0")
    return plt
end

function create_convergence_plt(f, g, x0, n, prob)
    optimizeHookeJeeves(f, g, x0, n, prob, plotType = 2)
end


p1 = create_rosenbrock_plt([2,2])
p2 = create_rosenbrock_plt([0,0])
p3 = create_rosenbrock_plt([4,3])
plt1 = plot(p1,p2,p3,layout=(3,1))
gui(plt1)

p4 = plot(overwrite_figure=false)
create_convergence_plt(rosenbrock,rosenbrock_gradient,[0,0],20,"simple_1")
title!("Rosenbrock function convergence")
ylabel!("f(x)")
p5 = plot(overwrite_figure=false)
create_convergence_plt(powell,powell_gradient,[0,0,0,0],40,"simple_2")
title!("Powell function convergence")
ylabel!("f(x)")
p6 = plot(overwrite_figure=false)
create_convergence_plt(himmelblau,himmelblau_gradient,[0,0],100,"simple_3")
title!("Himmelblau function convergence")
ylabel!("f(x)")
xlabel!("Number of iterations")
plt2 = plot(p4,p5,p6,layout=(3,1))
gui(plt2)
