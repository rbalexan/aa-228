function chooseAction(p::MultiFareDynamicPricingProblem, Q::Array, sLinearIndex::Int, solver::Symbol)

    # Initialize action space
    actionSpace, 𝖠  = actionSpaceAttributes(p)

    if solver == :sarsa || solver == :sarsaLambda # Choose an action using the ϵ-greedy algorithm
        aLinearIndex = rand() <= p.ϵ ? rand(1:𝖠) : argmax(Q[sLinearIndex, :])
        a            = aLinearIndexToRaw(p, aLinearIndex)
    elseif solver == :random # Choose a random action
        aLinearIndex = rand(1:𝖠)
        a            = aLinearIndexToRaw(p, aLinearIndex)
    elseif solver == :staticLow # Choose the minimum prices for all fare classes
        a            = Dict(f => min(p.fareClasses[f].fareActionSpace...) for f in keys(p.fareClasses))
        aLinearIndex = aRawToLinearIndex(p, a)
    elseif solver == :staticHigh # Choose the maximum prices for all fare classes
        a            = Dict(f => max(p.fareClasses[f].fareActionSpace...) for f in keys(p.fareClasses))
        aLinearIndex = aRawToLinearIndex(p, a)
    end

    # Return the action and its LinearIndex representation
    return a, aLinearIndex
end
