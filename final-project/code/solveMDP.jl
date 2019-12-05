function solveMDP(p::multiFareDynamicPricingProblem)

    # Initialize customer lists
    customersWithoutTickets = Dict(f => Set() for f in keys(fareClasses))
    customersWithTickets    = Dict(f => Set() for f in keys(fareClasses))

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1
    @show "Start", ticketsAvailable, t
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
    a, aLinearIndex = chooseAction(p, Q, 𝖠_space, sLinearIndex)
    @show "Starting action", a, aLinearIndex
    # Loop in t

    for t = 1:p.timeHorizon
        @show "NEW LOOP------------------------------------------------------------", t
        ticketsSold′          = Dict(f => 0     for f in keys(p.fareClasses))
        customersWithPurchase = Dict(f => Set() for f in keys(p.fareClasses))
        newCustomers          = Dict(f => 0     for f in keys(p.fareClasses))

        for f in keys(p.fareClasses)
            _, ticketsSold′[f], _, newCustomers[f], customersWithoutTickets[f], customersWithPurchase[f] = generativeModel(problem, f, ticketsAvailable, t, a[f], customersWithoutTickets[f])
        @show "Ticket demand", ticketsSold′[f], newCustomers[f]
        end

        if sum([ticketsSold′[f] for f in keys(p.fareClasses)]) > ticketsAvailable
        @show "DEMAND > AVAILABILITY"
            # Filter customersWithPurchase so that its length is exactly equal to ticketsAvailable
            customersWithPurchaseAll = [(f,customer) for f in keys(p.fareClasses) for customer in customersWithPurchase[f]]
        @show "Demand count", length(customersWithPurchaseAll)
            customersWithPurchaseAll = shuffle(customersWithPurchaseAll)[1:ticketsAvailable]
            customersWithPurchase = Dict(f => Set([c[2] for c in filter(x->x[1]==f,customersWithPurchaseAll)]) for f in keys(p.fareClasses))
        @show "Tickets sold", length(customersWithPurchase[:mixed]), length(customersWithPurchase[:leisure]), length(customersWithPurchase[:business])
            #Update ticketsSold′
            ticketsSold′ = Dict(f => length(customersWithPurchase[f]) for f in keys(p.fareClasses))
        @show "Tickets sold dict", ticketsSold′
        end

        @show "Tickets sold", [length(customersWithPurchase[f]) for f in keys(p.fareClasses)]
        @show "Notix", [length(customersWithoutTickets[f]) for f in keys(p.fareClasses)]
        @show "Tix", [length(customersWithTickets[f]) for f in keys(p.fareClasses)]
        #  Process purchases
        for f in keys(p.fareClasses)
        @show "PROCESSING PURCHASES"
            setdiff!(customersWithoutTickets[f], customersWithPurchase[f])
            union!(  customersWithTickets[f],    customersWithPurchase[f])
        end
        @show "Notix", [length(customersWithoutTickets[f]) for f in keys(p.fareClasses)]
        @show "Tix", [length(customersWithTickets[f]) for f in keys(p.fareClasses)]

        # Calculate new state and reward
        ticketsAvailable′ = ticketsAvailable - sum([ticketsSold′[f] for f in keys(p.fareClasses)])
        t′ = t + 1
        r = sum([a[f]*ticketsSold′[f] for f in keys(p.fareClasses)])
        @show "New state and reward", ticketsAvailable′, t′, r

        # Choose next action
        sCartesianIndex′ = CartesianIndex(findfirst(x->x==ticketsAvailable′,0:p.totalTickets),findfirst(x->x==t′,1:p.timeHorizon))
        sLinearIndex′ = LinearIndices(𝖲_space)[sCartesianIndex′]
        a′, aLinearIndex′ = chooseAction(p, Q, 𝖠_space, sLinearIndex′)

        @show "new action", a′, aLinearIndex′

        # Implement the Sarsa update step
        @show "SARSA update"
        @show "Old value", Q[sLinearIndex,  aLinearIndex]
        Q[sLinearIndex,  aLinearIndex] += p.η*(r + p.γ*Q[sLinearIndex′,  aLinearIndex′] - Q[sLinearIndex,  aLinearIndex])
        @show "Primed value", Q[sLinearIndex′,  aLinearIndex′]
        @show "New value", Q[sLinearIndex,  aLinearIndex]

        # Update state and action
        ticketsAvailable = ticketsAvailable′
        t = t′
        sLinearIndex = sLinearIndex′

        a = a′
        aLinearIndex,  aLinearIndex′

        @show "Tickets available", ticketsAvailable

        if ticketsAvailable <= 0
            break
        end
    end

    # Extract policy
    #π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    #π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
