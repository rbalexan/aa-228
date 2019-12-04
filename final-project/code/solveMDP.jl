function solveMDP(p::multiFareDynamicPricingProblem)

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1

    # Initialize state and action spaces

    𝖲_space = LinearIndices((0:p.totalTickets,1:p.timeHorizon))
    𝖲_size = length(𝖲_space)
    sCartesianIndex = CartesianIndex(findfirst(x->x==ticketsAvailable,0:p.totalTickets),findfirst(x->x==t,1:p.timeHorizon))
    sLinearIndex = LinearIndices(x)[s_index]

    𝖠_space = LinearIndices(zeros([length(p.fareClasses[f].actionSpace) for f in keys(p.fareClasses)]...))
    𝖠_size = length(𝖠_space)

    # Initialize Q and r
    Q = zeros(𝖲_size, 𝖠_size)
    r = 0

    # Choose action

    function chooseAction(p::multiFareDynamicPricingProblem, 𝖠_space::LinearIndices, LinearIndex::Int, 𝖲_size::Int)
        ϵ_gaussian      = rand(Normal(p.ϵ, 0))
        aLinearIndex    = rand(Bernoulli(ϵ_gaussian)) == 1 ? rand(1:𝖲_size) : argmax(Q[sLinearIndex, :])
        aCartesianIndex = CartesianIndices(𝖠_space)[aLinearIndex]
        a               = Dict(f => p.fareClasses[f].actionSpace[aCartesianIndex[i]] for (i,f) in enumerate(keys(p.fareClasses)))
        return a, aLinearIndex
    end

    a, aLinearIndex = chooseAction(p, 𝖠_space, sLinearIndex, 𝖲_size)

    # Loop in t

    for t = 1:p.timeHorizon

        ticketsSold′          = Dict(k => Set() for k in keys(p.fareClasses))
        customersWithPurchase = Dict(k => Set() for k in keys(p.fareClasses))

        for f in keys(p.fareClasses)
            _, ticketsSold′[f], _, customersWithPurchase[f] = generativeModel(problem, f, ticketsAvailable, t, a[f])
        end
        if sum([ticketsSold′[f] for f in keys(p.fareClasses)]) > ticketsAvailable
            # Filter customersWithPurchase so that its length is exactly equal to ticketsAvailable
            customersWithPurchase = shuffle(customersWithPurchase)[1:ticketsAvailable]
            #Update ticketsSold′
            ticketsSold′ = ticketsAvailable
        end

        #  Process purchases
        for f in keys(p.fareClasses)
            setdiff!(customersWithoutTickets[f], customersWithPurchase[f])
            union!(  customersWithTickets[f],    customersWithPurchase[f])
        end

        # Calculate new state and reward
        ticketsAvailable′ = ticketsAvailable - sum([ticketsSold′[f] for f in keys(p.fareClasses)])
        t′ = t + 1
        r = sum([a[f]*ticketsSold[f] for f in keys(p.fareClasses)])

        # Choose next action
        sCartesianIndex′ = CartesianIndex(findfirst(x->x==ticketsAvailable′,0:p.totalTickets),findfirst(x->x==t′,1:p.timeHorizon))
        sLinearIndex′ = LinearIndices(x)[s_index]
        a′, aLinearIndex′ = chooseAction(p, 𝖠_space, sLinearIndex′, 𝖲_size)

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
    π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
