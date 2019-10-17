using LightGraphs
using DataFrames
using GraphPlot
using Compose
using SpecialFunctions
using CSV
using Printf

include("bayesianScore.jl")
include("bayesianCounts.jl")
include("initializeExampleGraph.jl")
include("initializeUnconnectedGraph.jl")
include("initializeRandomGraph.jl")
include("writeGraph.jl")

infile  = ARGS[1]
outfile = ARGS[2]

# load the dataset
dataframe = CSV.read("data/" * infile * ".csv")

# initialize graph
graph = initializeUnconnectedGraph(n)
#graph = initializeExampleGraph()
#graph = initializeRandomGraph(n,trunc(Int, rand()*n^1.5))

# compute the Bayesian counts and Bayesian score of the graph
n, q, r, m  = bayesianCounts(graph)
score       = bayesianScore(n, q, r, m)

@show score

# output the .gph and .svg files
nodes = Dict(i => names(df)[i] for i in 1:n)
writeGraph(graph, nodes, "graphs/" * outfile * ".gph")
draw(SVG("graphs/" * outfile * ".svg", 16cm, 16cm), gplot(graph, nodesize=0.5))

# between search iterations, mijk counts only change slightly
# nodes with changes in parental structures
