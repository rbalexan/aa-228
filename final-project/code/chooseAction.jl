function chooseAction(p::multiFareDynamicPricingProblem, Q::Array, actionSpace::LinearIndices, sLinearIndex::Int)
    aLinearIndex    = rand() <= p.ϵ ? rand(1:length(actionSpace)) : argmax(Q[sLinearIndex, :])
    aCartesianIndex = CartesianIndices(actionSpace)[aLinearIndex]
    a               = Dict(f => p.fareClasses[f].fareActionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))
    return a, aLinearIndex
end
