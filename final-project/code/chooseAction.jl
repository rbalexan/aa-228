function chooseAction(p::multiFareDynamicPricingProblem, Q::Array, actionSpace::LinearIndices, sLinearIndex::Int)
    ϵ_gaussian      = rand(Normal(p.ϵ, 0))
    aLinearIndex    = rand() <= ϵ_gaussian ? rand(1:length(actionSpace)) : argmax(Q[sLinearIndex, :])
    aCartesianIndex = CartesianIndices(actionSpace)[aLinearIndex]
    a               = Dict(f => p.fareClasses[f].fareActionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))
    return a, aLinearIndex
end
