function chooseAction(p::MultiFareDynamicPricingProblem, Q::Array, sLinearIndex::Int)

    # Initialize action space
    actionSpace, 𝖠  = actionSpaceAttributes(p)

    # Choose an action using the ϵ-greedy algorithm
    aLinearIndex    = rand() <= p.ϵ ? rand(1:𝖠) : argmax(Q[sLinearIndex, :])

    # Format the action
    aCartesianIndex = CartesianIndices(actionSpace)[aLinearIndex]
    a               = Dict(f => p.fareClasses[f].fareActionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))

    # Return the action and its linear index representation
    return a, aLinearIndex

end
