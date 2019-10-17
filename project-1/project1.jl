using LightGraphs
using DataFrames
using TikzGraphs
using TikzPictures
using SpecialFunctions
using CSV
using Printf

include("bayesianScore.jl")
include("initializeExampleGraph.jl")
include("initializeRandomGraph.jl")
include("writeGraph.jl")

function compute(infile, outfile)

    # s         - samples (rows)
    # n         - random variables (columns)
    # qi        - number of parental instantiations of Xi
    # ri        - number of instantiations of Xi
    # i ∈ 1:n   - random variable index
    # j ∈ 1:qi  - parental instantiation index
    # k ∈ 1:ri  - random variable value index

    dataframe = CSV.read("data/" * infile)
    data = Matrix(dataframe)
    s, n = size(data)
    idx2names = Dict(i => names(dataframe)[i] for i in 1:n)

    global r = [maximum(data[:, i]) for i in 1:n]

    # initialize graph
    graph = initializeExampleGraph()
    #graph = initializeRandomGraph(n,trunc(Int, rand()*n^1.5))

    writeGraph(graph, idx2names, "graphs/" * outfile)

    global parents = Dict(i => sort(inneighbors(graph, i)) for i in 1:n)
    global q = [prod(r[parents[i]]) for i in 1:n]
    global m = Dict()

    for s in 1:s        # loop over all samples (rows)
        for i in 1:n     # loop over all Xis

            # look up value assignment of Xi (k)
            k = data[s, i]

            if q[i] == 1    # if Xi has no parents
                j = 1
            else            # if Xi has parents

                # get sample values and number of possible instantiations of each parent
                parentalValues = Tuple(data[s, parents[i]])
                parentalInstantiations = Tuple(r[parents[i]])

                j = LinearIndices(parentalInstantiations)[CartesianIndex(parentalValues)]

            end

            # safely establish and increment mijk
            m[(i, j, k)] = get(m, (i, j, k), 0) + 1

        end
    end

    score = bayesianScore(n, q, r, m)
    @show score

    # between search iterations, mijk counts only change slightly
    # nodes with changes in parental structures

end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
