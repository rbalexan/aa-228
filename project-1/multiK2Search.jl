using LightGraphs
using DataFrames
using GraphPlot
using Compose
using SpecialFunctions
using Random
using CSV
using Printf
using Colors

include("bayesianScore.jl")
include("bayesianCounts.jl")
include("initializeExampleGraph.jl")
include("initializeUnconnectedGraph.jl")
include("singleK2Search.jl")
include("writeGraph.jl")

iterations  = parse(Int64, ARGS[1])
infile      = ARGS[2]
outfile     = ARGS[3]

Random.seed!(2)

function multiK2Search(dataset, iterations)

    bestScore = -1E20
    bestGraph = DiGraph(0)

    for iteration in 1:iterations

        @show iteration

        graph, score = singleK2Search(dataset)

        if score > bestScore
            bestScore = score
            bestGraph = graph
        end

        @show bestScore

    end

    return bestGraph, bestScore

end

# load the dataset
dataset = CSV.read("data/" * infile * ".csv")

(bestGraph, bestScore), time = @timed multiK2Search(dataset, iterations)
@show time
@show bestScore

# output the .gph and .svg files
println("printing graph")
nodes = Dict(i => names(dataset)[i] for i in 1:size(dataset)[2])
writeGraph(bestGraph, nodes, "graphs/" * outfile * ".gph")
draw(SVG("graphs/" * outfile * ".svg", 16cm, 16cm), gplot(bestGraph,
    nodelabel=names(dataset), nodefillc=[RGB(0.10, 0.58, 0.86) for i in 1:size(dataset)[2]]))
open("graphs/" * outfile * ".dat", "w") do io
    @printf(io, "Best Score | %f \n", bestScore)
    @printf(io, "Time       | %f \n", time)
    @printf(io, "Iterations | %.0f", iterations)
end
