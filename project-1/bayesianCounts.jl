function bayesianCounts(dataframe::DataFrame, graph::DiGraph)

    # s         - samples (rows)
    # n         - random variables (columns)
    # qi        - number of parental instantiations of Xi
    # ri        - number of instantiations of Xi
    # i ∈ 1:n   - random variable index
    # j ∈ 1:qi  - parental instantiation index
    # k ∈ 1:ri  - random variable value index

    data = Matrix(dataframe)
    s, n = size(data)

    global parents = Dict(i => sort(inneighbors(graph, i)) for i in 1:n)
    global r = [maximum(data[:, i]) for i in 1:n]
    global q = [prod(r[parents[i]]) for i in 1:n]

    global m = Dict()

    for s in 1:s, i in 1:n  # loop over all samples (rows), loop over all Xis

        # look up value assignment of Xi (k)
        k = data[s, i]

        if q[i] == 1 # if Xi has no parents

            j = 1

        else # if Xi has parents

            # compute the parental instantiation index (j)
            parentalValues = Tuple(data[s, parents[i]])
            parentalInstantiations = Tuple(r[parents[i]])

            j = LinearIndices(parentalInstantiations)[CartesianIndex(parentalValues)]

        end

        # safely establish and increment mijk
        m[(i, j, k)] = get(m, (i, j, k), 0) + 1

    end

    return n, q, r, m

end
