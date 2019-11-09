function valueIteration(𝖲::Int, 𝖠::Int, dataset::DataFrame, reachableStates,
    γ::Float64, ϵ::Float64)

    T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)

    # compute and show the Bellman residual
    δ = ϵ*(1-γ)/γ
    bellmanResidual = δ+1
    @show δ

    U  = zeros(𝖲)
    Up = zeros(𝖲)
    π  = zeros(𝖲)

    sumOfDiscountedFutureRewards = zeros(𝖲, 𝖠)
    immediateReward              = zeros(𝖲, 𝖠)

    # initialize immediate reward matrix
    for s in 1:𝖲, a in 1:𝖠
        immediateReward[s, a] = get(R, (s, a), 0)
    end

    k = 1

    while bellmanResidual > δ

        sumOfDiscountedFutureRewards = zeros(𝖲, 𝖠)

        for s in 1:𝖲, a in 1:𝖠

            sumOfDiscountedFutureRewards[s, a] = γ*sum(get(T, (s, a, sp), 0)*Up[sp] for sp in reachableStates(s))

        end

        # update value function and policy over the entire state space
        Up, π = findmax(immediateReward + sumOfDiscountedFutureRewards, dims=2)
        bellmanResidual = maximum(abs.(Up - U))

        @show k
        @show bellmanResidual

        k += 1
        U = deepcopy(Up)

    end

    return Up, collect(π[i][2] for i in 1:𝖲)

end
