function solveMDP(p::multiFareDynamicPricingProblem)

    # Initialize state = (ticketsAvailable, time)
    ticketsAvailable = p.totalTickets
    time = 0

    # Initialize Q
    𝖲 = p.totalTickets * p.timeHorizon
    𝖠 = prod([length(p.fareClasses[f].actionSpace) for f in 1:length(p.fareClasses)])
    Q = zeros(𝖲, 𝖠)

    # Initialize reward
    r = 0

    s = LinearIndices((1:p.totalTickets, 1:p.timeHorizon))[ticketsAvailable, time] # may need to change the linear indexing
    ϵ_gaussian = rand(Normal(p.ϵ, 0), 1)[]
    a = rand(Bernoulli(ϵ_gaussian), 1)[] == 1 ? rand(1:𝖲, 1) : argmax(Q[s, :])

    # Loop along time
    # *Code to be added*\
    # sarsa

    # Extract policy
    π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
