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

"""
    writeGraph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function writeGraph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s, %s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function compute(infile, outfile)

    dataframe = CSV.read("data/" * infile)
    data = Matrix(dataframe)

    # i 1:n  - random variables
    # j 1:qi - instantiations of the parents
    # k 1:ri - values of Xi

    # m samples (rows), n random variables (columns)
    m, n = size(data)

    r = zeros(Int64, 1, n)
    q = zeros(Int64, 1, n)
    kDict = Dict()
    parentsDict = Dict()
    idx2NamesDict = Dict()

    for i in 1:n

        idx2NamesDict[i] = names(dataframe)[i]

        uniqueInstantiations = unique(data[:, i])
        # compute ri (number of unique instantiations of Xi)
        global r[i] = size(uniqueInstantiations)[1]

        # index k for each Xi (assign indices to each unique instantiation of Xi)
        global kDict[i] = sort(uniqueInstantiations)

    end

    # initialize graph DiGraph (right now its equal to example
    # later it will be based on search strategy)
    graph = initializeExampleGraph()
    #graph = initializeRandomGraph(n)

    writeGraph(graph, idx2NamesDict, "graphs/" * outfile)

    for i in 1:n

        # compute qi from graph structure
        parents = sort(inneighbors(graph, i))

        global parentsDict[i] = parents

        prod = 1
        for j = 1:length(parents)
            ri = r[parents[j]]
            prod *= ri
        end

        global q[i] = prod

    end

    global mDict = Dict()
    global αDict = Dict()

    # loop over all samples (rows)
    for m in 1:m

        # loop over all Xis
        for i = 1:n

            # look up value assignment of Xi (k)
            k = data[m, i]

            if q[i] == 1
            # if Xi has no parents, j is 1
                j = 1
            else

                parents = parentsDict[i]
                numParentalInstantiations = r[parents]
                parentalInstantiationArray = LinearIndices(zeros(tuple(numParentalInstantiations...)))

                parentalValues = data[m, parents]

                # indices = zeros((1, q[i]))
                # for w in 1:q[i]
                #     kValues = kDict(w)
                #     global indices = findall(x->x==parentValues[w],kValues)
                #
                # j = parentalInstantiationArray[#k-index for parent 1]

                # !!this is only if the values of all vars start at one go up from there (no skipping)
                j = getindex(parentalInstantiationArray, parentalValues)[1]
            end

            mDict[(i, j, k)] = get(mDict, (i, j, k), 0) + 1

        end

    end

    score = bayesianScore(n, q, r, mDict)
    @show score
    # between search iterations, mijk counts only change slightly
    # nodes with changes in parental structures

end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
