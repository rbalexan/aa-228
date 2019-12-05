function solveMDP(p::multiFareDynamicPricingProblem, stateSpace::LinearIndices, actionSpace::LinearIndices, 𝖲::Int, 𝖠::Int, Q::Array)

    # Initialize customer lists
    customersWithoutTickets = Dict(f => Set() for f in keys(fareClasses))
    customersWithTickets    = Dict(f => Set() for f in keys(fareClasses))

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1
    @show "Start", ticketsAvailable, t
    sCartesianIndex = CartesianIndex(findfirst(x->x==ticketsAvailable, 0:p.totalTickets), findfirst(x->x==t, 1:p.timeHorizon))
    sLinearIndex    = LinearIndices(stateSpace)[sCartesianIndex]

    # Choose action
    a, aLinearIndex = chooseAction(p, Q, actionSpace, sLinearIndex)
    @show "Starting action", a, aLinearIndex
    # Loop in t

    ticketsSold           = Dict(f => 0     for f in keys(p.fareClasses))
    customersWithPurchase = Dict(f => Set() for f in keys(p.fareClasses))
    newCustomers          = Dict(f => 0     for f in keys(p.fareClasses))

# customersWith/OutTickets is permanent storage

    for t in 1:p.timeHorizon
        @show "NEW LOOP------------------------------------------------------------", t

        # move [f] arguments into function, and return
        for f in keys(p.fareClasses)
            _, ticketsSold[f], _, newCustomers[f], customersWithoutTickets[f], customersWithPurchase[f] = generativeModel(problem, f, ticketsAvailable, t, a[f], customersWithoutTickets[f])
            @show "Ticket demand", ticketsSold[f], newCustomers[f]
        end

        # if all tickets (over)sold
        if sum([ticketsSold[f] for f in keys(p.fareClasses)]) > ticketsAvailable
            @show "DEMAND > AVAILABILITY"
            # Filter customersWithPurchase so that its length is exactly equal to ticketsAvailable
            customersWithPurchaseAll = [(f, customer) for f in keys(p.fareClasses) for customer in customersWithPurchase[f]]
            @show "Demand count", length(customersWithPurchaseAll)
            customersWithPurchaseAll = shuffle(customersWithPurchaseAll)[1:ticketsAvailable] # grab random customers to properly fill tickets
            customersWithPurchase = Dict(f => Set([c[2] for c in filter(x->x[1]==f,customersWithPurchaseAll)]) for f in keys(p.fareClasses))
            # add comprehension for show macro
            @show "Tickets sold", length(customersWithPurchase[:mixed]), length(customersWithPurchase[:leisure]), length(customersWithPurchase[:business])
            #Update ticketsSold
            ticketsSold = Dict(f => length(customersWithPurchase[f]) for f in keys(p.fareClasses))
            @show "Tickets sold dict", ticketsSold
        end

        @show "Tickets sold", [length(customersWithPurchase[f])     for f in keys(p.fareClasses)]
        @show "Notix",        [length(customersWithoutTickets[f])   for f in keys(p.fareClasses)]
        @show "Tix",          [length(customersWithTickets[f])      for f in keys(p.fareClasses)]

        #  Process purchases
        @show "PROCESSING PURCHASES"
        for f in keys(p.fareClasses)
            setdiff!(customersWithoutTickets[f], customersWithPurchase[f])
            union!(  customersWithTickets[f],    customersWithPurchase[f])
        end
        @show "Notix",  [length(customersWithoutTickets[f]) for f in keys(p.fareClasses)]
        @show "Tix",    [length(customersWithTickets[f])    for f in keys(p.fareClasses)]

        # Calculate new state and reward
        ticketsAvailable′ = ticketsAvailable - sum([ticketsSold[f] for f in keys(p.fareClasses)])
        t′ = t + 1 # not used, but nice to make sure :)
        r = sum([a[f]*ticketsSold[f] for f in keys(p.fareClasses)])
        @show "New state and reward", ticketsAvailable′, t′, r

        # Break if all tickets sold or time is up
        if t == p.timeHorizon || ticketsAvailable′ <= 0
            break
        end

        # Choose next action
        sCartesianIndex′    = CartesianIndex(findfirst(x->x==ticketsAvailable′,0:p.totalTickets),findfirst(x->x==t′,1:p.timeHorizon))
        sLinearIndex′       = LinearIndices(stateSpace)[sCartesianIndex′]
        a′, aLinearIndex′   = chooseAction(p, Q, actionSpace, sLinearIndex′)
        @show "new action", a′, aLinearIndex′

        # Implement the Sarsa update step
        # logic for MDP solver (SARSA or SARSA(λ))
        @show "SARSA update"
        @show "Old value",     Q[sLinearIndex, aLinearIndex]
        Q[sLinearIndex,  aLinearIndex] += p.η*(r + p.γ * Q[sLinearIndex′, aLinearIndex′] - Q[sLinearIndex, aLinearIndex])
        @show "Primed value",  Q[sLinearIndex′, aLinearIndex′]
        @show "Updated value", Q[sLinearIndex, aLinearIndex]
        @show size(Q)

        # Update state and action
        ticketsAvailable = ticketsAvailable′
        t                = t′
        a                = a′
        sLinearIndex     = sLinearIndex′
        aLinearIndex     = aLinearIndex′

        @show "Tickets available", ticketsAvailable

    end

    return Q, r

end
