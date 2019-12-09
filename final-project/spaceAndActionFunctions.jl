# Get state and action spaces
stateSpaceAttributes( p::MultiFareDynamicPricingProblem) = [(x, length(x)) for x in [LinearIndices((0:p.totalTickets,1:p.timeHorizon))]][1]
actionSpaceAttributes(p::MultiFareDynamicPricingProblem) = [(x, length(x)) for x in [LinearIndices(zeros([length(p.fareClasses[f].fareActionSpace) for f in keys(p.fareClasses)]...))]][1]

# Convert states and actions from their raw forms, i.e., s = (ticketsAvailable, t) and
# a = (fare1Price, fare2Price, fare3Price, ...), which are vectors, into their LinearIndex
# forms, i.e., a scalar

function sRawToLinearIndex(p::MultiFareDynamicPricingProblem, ticketsAvailable::Int, t::Int)

   stateSpace, 𝖲        = stateSpaceAttributes(p)
   stateComponentSpaces = [collect(0:p.totalTickets), collect(1:p.timeHorizon)]
   sCartesianIndex      = CartesianIndex([findfirst(x->x==stateComponent,stateComponentSpace)
                           for (stateComponent, stateComponentSpace) in zip([ticketsAvailable,t], stateComponentSpaces)]...)
   sLinearIndex         = LinearIndices(stateSpace)[sCartesianIndex]

   return sLinearIndex
end

function aRawToLinearIndex(p::MultiFareDynamicPricingProblem, a::Dict)

   actionSpace, 𝖠        = actionSpaceAttributes(p)
   fareActionSpaces      = [p.fareClasses[f].fareActionSpace for f in keys(p.fareClasses)]
   aCartesianIndex       = CartesianIndex([findfirst(x->x==fareAction,fareActionSpace)
                           for (fareAction, fareActionSpace) in zip([a[f] for f in keys(a)], fareActionSpaces)]...)
   aLinearIndex          = LinearIndices(actionSpace)[aCartesianIndex]

   return aLinearIndex
end

function sLinearIndexToRaw(p::MultiFareDynamicPricingProblem, sLinearIndex::Int)

   stateSpace, 𝖲    = stateSpaceAttributes(p)
   sCartesianIndex  = CartesianIndices(stateSpace)[sLinearIndex]
   ticketsAvailable = collect(0:p.totalTickets)[sCartesianIndex[1]]
   t                = collect(1:p.timeHorizon)[sCartesianIndex[2]]

   return ticketsAvailable, t
end

function aLinearIndexToRaw(p::MultiFareDynamicPricingProblem, aLinearIndex::Int)

   actionSpace, 𝖠  = actionSpaceAttributes(p)
   aCartesianIndex = CartesianIndices(actionSpace)[aLinearIndex]
   a               = Dict(f => p.fareClasses[f].fareActionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))

   return a
end
