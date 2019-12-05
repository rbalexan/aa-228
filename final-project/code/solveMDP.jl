function solveMDP(p::multiFareDynamicPricingProblem)

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1

    # Initialize state and action spaces

    𝖲_space = LinearIndices((0:p.totalTickets,1:p.timeHorizon))
    𝖲_size = length(𝖲_space)
    sCartesianIndex = CartesianIndex(findfirst(x->x==ticketsAvailable,0:p.totalTickets),findfirst(x->x==t,1:p.timeHorizon))
    sLinearIndex = LinearIndices(𝖲_space)[sCartesianIndex]

    𝖠_space = LinearIndices(zeros([length(p.fareClasses[f].actionSpace) for f in keys(p.fareClasses)]...))

    # Initialize Q and r
    Q = zeros(𝖲_size, 𝖠_size)
    r = 0

    # Choose action

    function chooseAction(p::multiFareDynamicPricingProblem, 𝖠_space::LinearIndices, sLinearIndex::Int)
        ϵ_gaussian      = rand(Normal(p.ϵ, 0))
        𝖠_size = length(𝖠_space)
        aLinearIndex    = rand() <= ϵ_gaussian ? rand(1:𝖠_size) : argmax(Q[sLinearIndex, :])
        aCartesianIndex = CartesianIndices(𝖠_space)[aLinearIndex]
        a               = Dict(f => p.fareClasses[f].actionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))
        return a, aLinearIndex
    end

    a, aLinearIndex = chooseAction(p, 𝖠_space, sLinearIndex)

    # Loop in t

    for t = 1:p.timeHorizon

        ticketsSold′          = Dict(f => 0     for f in keys(p.fareClasses))
        customersWithPurchase = Dict(f => Set() for f in keys(p.fareClasses))
        @show t, ticketsSold′, customersWithPurchase

        for f in keys(p.fareClasses)
            _, ticketsSold′[f], _, customersWithPurchase[f] = generativeModel(problem, f, ticketsAvailable, t, a[f])
        end
        if sum([ticketsSold′[f] for f in keys(p.fareClasses)]) > ticketsAvailable
            # Filter customersWithPurchase so that its length is exactly equal to ticketsAvailable
            customersWithPurchaseAll = [(f,customer) for f in keys(p.fareClasses) for customer in customersWithPurchase[f]]
            @show length(customersWithPurchaseAll)
            customersWithPurchaseAll = shuffle(customersWithPurchaseAll)[1:ticketsAvailable]
            customersWithPurchase = Dict(f => Set([c[2] for c in filter(x->x[1]==f,customersWithPurchaseAll)]) for f in keys(p.fareClasses))
            @show length(customersWithPurchase[:mixed]), length(customersWithPurchase[:leisure]), length(customersWithPurchase[:business])
            #Update ticketsSold′
            ticketsSold′ = Dict(f => length(customersWithPurchase[f]) for f in keys(p.fareClasses))
            @show ticketsSold′
        end

        #  Process purchases
        for f in keys(p.fareClasses)
            setdiff!(customersWithoutTickets[f], customersWithPurchase[f])
            union!(  customersWithTickets[f],    customersWithPurchase[f])
        end

        # Calculate new state and reward
        ticketsAvailable′ = ticketsAvailable - sum([ticketsSold′[f] for f in keys(p.fareClasses)])
        t′ = t + 1
        r = sum([a[f]*ticketsSold′[f] for f in keys(p.fareClasses)])

        # Choose next action
        sCartesianIndex′ = CartesianIndex(findfirst(x->x==ticketsAvailable′,0:p.totalTickets),findfirst(x->x==t′,1:p.timeHorizon))
        sLinearIndex′ = LinearIndices(𝖲_space)[sCartesianIndex′]
        a′, aLinearIndex′ = chooseAction(p, 𝖠_space, sLinearIndex′)

        # Implement the Sarsa update step
        Q[sLinearIndex,  aLinearIndex] += p.η*(r + p.γ*Q[sLinearIndex′,  aLinearIndex′] - Q[sLinearIndex,  aLinearIndex])

        # Update state and action
        ticketsAvailable = ticketsAvailable′
        t = t′
        a = a′

        if ticketsAvailable′ == 0
            break
        end
    end

    # Extract policy
    #π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    #π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
