module AA222Project1_2019

using JSON
using PyCall
using Random
using Statistics

const project1 = PyNULL()
const err = Ref{String}("")


function append_to_python_search_path(str::AbstractString)
    pushfirst!(PyVector(pyimport("sys")."path"), str)
end

const COUNTERS = Dict{String, Int}()
const PROBS = Dict{String, Dict{Int, Any}}()

"""
Count the number of times a function was evaluated
"""
macro counted(f)
    name = f.args[1].args[1]
    name_str = String(name)
    body = f.args[2]
    update_counter = quote
        if !haskey(COUNTERS, $name_str)
            COUNTERS[$name_str] = 0
        end
        COUNTERS[$name_str] += 1
    end
    insert!(body.args, 1, update_counter)
    return f
end

"""
    get_score(f, g, x_star_hat, n)

Arguments
    - `f`: function
    - `g`: gradient of function
    - `x_star_hat`: Found optima
    - `n`: Number of evaluations allowed (f+2g)
Returns
    - `num_evals`
    - `score`
"""
function get_score(f, g, x_star_hat, n)
    f_name = String(typeof(f).name.mt.name)
    g_name = String(typeof(g).name.mt.name)

    if haskey(COUNTERS, f_name)
        f_evals = COUNTERS[f_name]
    else
        f_evals = 0
    end

    if haskey(COUNTERS, g_name)
        g_evals = COUNTERS[g_name]
    else
        g_evals = 0
    end

    num_evals = (f_evals + 2*g_evals)

    if num_evals <= n
        score = -f(x_star_hat)
        COUNTERS[f_name] -= 1
    else
        score = typemin(Int32)
    end
    return num_evals, score
end

if isfile(joinpath(@__DIR__, "simple.jl"))
    include(joinpath(@__DIR__, "simple.jl"))
    PROBS["simple"] = simple_problems
end


"""
    main(mode, pidx, repeat)

Arguments:
    - `mode`: simple
    - `pidx`: problem index
    - `repeat`: Number of Monte Carlo evaluations
Returns:
    - nothing
"""
function main(mode::String, pidx::Int, repeat::Int)
    if mode ∉ ["simple", "secret"]
        err[] = "Invalid mode = $(mode)"
    end

    inseed = 1337
    scores = Float64[]
    errors = String[]
    nevals = Int[]
    score = 0
    neval = 0
    if err[] == ""
        # Repeat the optimization with a different initialization
        for i in 1:repeat
            Random.seed!(inseed + i)
            prob = PROBS[mode][pidx]
            try
                if project1 != PyNULL()
                    opt_func = project1.optimize
                else
                    opt_func = optimize
                end
                x_star_hat = opt_func(prob.f, prob.g, prob.x0(), prob.n, "$(mode)_$(pidx)")
                neval, score = get_score(prob.f, prob.g, x_star_hat, prob.n)
                push!(errors, "")
            catch e
                push!(errors, "$(e)")
                x_star_hat = typemin(Int32)
                score = typemin(Int32)
                neval = typemin(Int32)
            end

            push!(nevals, neval)
            push!(scores, score)

            for k in keys(COUNTERS)
                delete!(COUNTERS, k)
            end
        end

        # So that we know if there was a problem with a particular seed
        enerrs = enumerate(errors)
        # Filter out empty
        enerrs = filter(x -> x[2] != "", collect(enerrs))
        err[] = join(enerrs, " \n")
    else
        push!(nevals, typemin(Int32))        
    end
    result = Dict("score" => mean(scores), "error"=> err[])

    open(".results_$(mode)_$(pidx).json", "w") do io
        write(io, JSON.json(result))
    end

    final_score = result["score"]
    max_evals = maximum(nevals)
    println("===================")
    println("Mode: $(mode)")
    println("Problem: $(pidx)")
    println("-------------------")
    if err[] != ""
        println("Error: $(err[])")
    end
    println("Max Num Function (and Gradient) Evals: $(max_evals)")
    println("Avg Score: $(final_score)")
    println("===================")
end

function __init__()
    if length(ARGS) < 1
        filepath = "project1.jl"
    else
        filepath = ARGS[1]
    end

    if isfile(filepath)
        if filepath[end-1:end] == "jl"
            @info("Including Julia file")
            try
                include(filepath)
            catch e
                err[] = "$(e)"
            end
        elseif filepath[end-1:end] == "py"
            @info("Importing Python file")
            try
                dir = dirname(filepath)
                append_to_python_search_path(dir)
                copy!(project1, pyimport("project1"))
            catch e
                err[] = "$(e)"
            end
        else
            err[] = "Not a correct file type"
        end
    else
        err[] = "File: $(filepath) doesn't exist"
    end
end

end # module
