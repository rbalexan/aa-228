function solveMDP(p::multiFareDynamicPricingProblem)

    # Initialize state = (ticketsAvailable, t)
    ticketsAvailable = p.totalTickets
    t = 1

    # Initialize Q
    𝖲 = p.totalTickets * p.timeHorizon
    𝖠 = prod([length(p.fareClasses[f].actionSpace) for f in keys(p.fareClasses)])
    Q = zeros(𝖲, 𝖠)

    # Initialize reward
    r = 0
    minTickets = 0
    minTime = 1

    s = LinearIndices((1:p.totalTickets, 1:p.timeHorizon))[ticketsAvailable - minTickets + 1, t - minTime + 1]
    ϵ_gaussian = rand(Normal(p.ϵ, 0), 1)[]
    a_index = rand(Bernoulli(ϵ_gaussian), 1)[] == 1 ? rand(1:𝖲, 1) : argmax(Q[s, :])
    a = ### TO ADD ### get the triplet prices from the a_index

    # Loop along time

    while true

        ticketsSold′          = Dict(k => Set() for k in keys(p.fareClasses))
        customersWithPurchase = Dict(k => Set() for k in keys(p.fareClasses))

        for f in keys(p.fareClasses)
            _, ticketsSold′[f], _, customersWithPurchase[f] = generativeModel(problem, f, ticketsAvailable, t, a)
        end
        if sum([ticketsSold′[f] for f in keys(p.fareClasses)]) > ticketsAvailable
            ### TO ADD ### Filter customersWithPurchase so that its length is exactly equal to ticketsAvailable
            ### TO ADD ### Update ticketsSold′
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
        ϵ_gaussian = rand(Normal(p.ϵ, 0), 1)[]
        a′_index = rand(Bernoulli(ϵ_gaussian), 1)[] == 1 ? rand(1:𝖲, 1) : argmax(Q[s, :])
        a′ = ### TO ADD ### get the triplet prices from the a′_index

        # Implement the Sarsa update step
        ### TO ADD ###

        # Update state and action
        ticketsAvailable = ticketsAvailable′
        t = t′
        a = a′

        if (t == p.timeHorizon) || (ticketsAvailable′ == 0)
            break
        end
    end

    # Extract policy
    π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
