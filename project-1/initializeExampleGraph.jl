function initializeExampleGraph()

    # initialize our example directed graph, G

    G = DiGraph(6)
    add_edge!(G, 1, 2)
    add_edge!(G, 3, 4)
    add_edge!(G, 5, 6)

    return G

end
