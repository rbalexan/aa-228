function solveMDP(p::Problem)

    # Initialize state
    v = p.V
    t = 0

    # Initialize Q
    𝖲 = p.V*p.T
    𝖠 = prod([length(p.F[f][7]) for f in 1:length(p.F)])
    Q = zeros(𝖲, 𝖠)

    # Initialize reward
    r = 0

    s_index    = LinearIndices((1:p.V, 1:p.T))[v, t] # may need to change the linear indexing
    ϵ_gaussian = rand(Normal(p.ϵ, 0), 1)[]
    a = rand(Bernoulli(ϵ_gaussian), 1)[] == 1 ? rand(1:300, 1) : argmax(Q[s_index, :])

    # Loop along time
    # *Code to be added*\

    # Extract policy
    π⋆ = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
    π⋆ = [π⋆[s][2] for s in 1:𝖲]
end
