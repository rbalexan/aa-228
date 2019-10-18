# K2 graph search algorithm (single iteration)
function singleK2Search(dataset)

    # initialize completely unconnected graph and score it
    graph       = initializeUnconnectedGraph(size(dataset)[2])
    score       = bayesianScore(bayesianCounts(dataset, graph)...)

    bestGraph   = deepcopy(graph)
    bestScore   = deepcopy(score)
    #@show bestScore

    # generate ordering of nodes
    nodeOrdering = randperm(size(dataset)[2]) # can remove the rng later
    @show nodeOrdering

    for i in nodeOrdering

        @show i

        # generate the set of possible parents for the node
        validParents = filter(x->x≠i, nodeOrdering)

        while length(inneighbors(graph, i)) < min(length(nodeOrdering) - 1, 8) # max parents

            # re/initialize possible graph scores
            graphScores = [-1.0E20 for i in 1:size(dataset)[2]]

            # remove new invalid graphs & compute the Bayesian score for all new valid graphs
            for j in deepcopy(validParents)

                add_edge!(graph, j, i)
                if is_cyclic(graph)
                    filter!(x->x≠j, validParents)
                else
                    graphScores[j] = bayesianScore(bayesianCounts(dataset, graph)...)
                end
                rem_edge!(graph, j, i)

            end

            # if we have a better graph, build, score, and save the best graph
            if maximum(graphScores) > bestScore

                bestParent = argmax(graphScores)
                add_edge!(graph, bestParent, i)

                bestGraph = deepcopy(graph)
                bestScore = maximum(graphScores)

                # remove best parent from list of possible parents
                filter!(x->x≠bestParent, validParents)

            else

                # no new graph yields an improvement
                break

            end

        end

        @show bestScore

    end

    return bestGraph, bestScore

end
