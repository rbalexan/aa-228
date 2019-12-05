function solveMDP(p::MultiFareDynamicPricingProblem, Q::Array)

    # Initialize state and action spaces
    stateSpace, 𝖲  = stateSpaceAttributes(p)
    actionSpace, 𝖠 = actionSpaceAttributes(p)

    # Initialize customer lists customersWithoutTickets and customersWithTickets - these are the
    # 'permanent' lists,as opposed to customersWithPurchase, which is updated at every time step
    customersWithoutTickets = Dict(f => Set() for f in keys(fareClasses))
    customersWithTickets    = Dict(f => Set() for f in keys(fareClasses))

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1
    @show "Start", ticketsAvailable, t
    sCartesianIndex = CartesianIndex(findfirst(x->x==ticketsAvailable, 0:p.totalTickets), findfirst(x->x==t, 1:p.timeHorizon))
    sLinearIndex    = LinearIndices(stateSpace)[sCartesianIndex]

    # Initialize total reward
    rTotal = 0

    # Choose action
    a, aLinearIndex = chooseAction(p, Q, sLinearIndex)
    @show "Starting action", a, aLinearIndex

    # Initialize other fare class-dependent variables
    ticketsSold           = Dict(f => 0     for f in keys(p.fareClasses))
    newCustomers          = Dict(f => 0     for f in keys(p.fareClasses))
    customersWithPurchase = Dict(f => Set() for f in keys(p.fareClasses))

    # Loop in time
    for t in 1:p.timeHorizon
        @show "NEW LOOP------------------------------------------------------------", t

        # Get customer demand to determine next state and reward
        for f in keys(p.fareClasses)
            _, ticketsSold[f], _, newCustomers[f], customersWithoutTickets[f], customersWithPurchase[f] =
                                        generativeModel(problem, f, ticketsAvailable, t, a, customersWithoutTickets)
            @show "Ticket demand", ticketsSold[f], newCustomers[f]
        end

        # Limit tickets to be sold if demand > availability
        if sum([ticketsSold[f] for f in keys(p.fareClasses)]) > ticketsAvailable
            @show "DEMAND > AVAILABILITY"

            # Filter customersWithPurchase (the list of customers making up the demand) so that its length is exactly equal to ticketsAvailable
            customersWithPurchaseAll = [(f, customer) for f in keys(p.fareClasses) for customer in customersWithPurchase[f]]
            customersWithPurchaseAll = shuffle(customersWithPurchaseAll)[1:ticketsAvailable] # grab random customers to properly fill tickets
            customersWithPurchase = Dict(f => Set([fareClassAndCustomer[2]
                    for fareClassAndCustomer in filter(x->x[1]==f, customersWithPurchaseAll)]) for f in keys(p.fareClasses))

            # Update ticketsSold
            ticketsSold = Dict(f => length(customersWithPurchase[f]) for f in keys(p.fareClasses))
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
        rTotal += r
        @show "New state and reward", ticketsAvailable′, t′, r

        # Break if time is up or all tickets are sold
        if t == p.timeHorizon || ticketsAvailable′ <= 0
            break
        end

        # Choose next action
        sCartesianIndex′    = CartesianIndex(findfirst(x->x==ticketsAvailable′,0:p.totalTickets),findfirst(x->x==t′,1:p.timeHorizon))
        sLinearIndex′       = LinearIndices(stateSpace)[sCartesianIndex′]
        a′, aLinearIndex′   = chooseAction(p, Q, sLinearIndex′)
        @show "new action", a′, aLinearIndex′

        # Implement the Sarsa update step
        #! Add logic for the type of MDP solver (SARSA or SARSA(λ))
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

    return Q, rTotal
end
