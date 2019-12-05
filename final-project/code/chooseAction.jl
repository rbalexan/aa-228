function chooseAction(p::multiFareDynamicPricingProblem, Q::Array, 𝖠_space::LinearIndices, sLinearIndex::Int)
    ϵ_gaussian      = rand(Normal(p.ϵ, 0))
    𝖠_size          = length(𝖠_space)
    aLinearIndex    = rand() <= ϵ_gaussian ? rand(1:𝖠_size) : argmax(Q[sLinearIndex, :])
    aCartesianIndex = CartesianIndices(𝖠_space)[aLinearIndex]
    a               = Dict(f => p.fareClasses[f].actionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))
    return a, aLinearIndex
end
