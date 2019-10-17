function initializeRandomGraph(n, nEdges)

    G = DiGraph(2)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 1)

    while is_cyclic(G)
        G = DiGraph(n, nEdges)
    end

    return G

end
