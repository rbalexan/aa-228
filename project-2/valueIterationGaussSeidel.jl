function valueIterationGaussSeidel(𝖲::Int, 𝖠::Int, dataset::DataFrame,
    γ::Float64, terminalStates, reachableStates, ϵ=1, reachableStateSpace=1:𝖲)

    T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)

    δ = ϵ*(1-γ)/γ
    bellmanResidual = δ+1
    @show δ

    U  = zeros(𝖲)
    Up = zeros(𝖲)
    π  = zeros(𝖲)

    sumOfDiscountedFutureRewards = zeros(𝖠)
    immediateReward              = zeros(𝖲, 𝖠)

    for s in 1:𝖲, a in 1:𝖠
        immediateReward[s, a] = get(R, (s, a), 0)
    end

    k = 1

    while bellmanResidual > δ

        for s in reachableStateSpace

            sumOfDiscountedFutureRewards = zeros(𝖠)

            for a in 1:𝖠

                if s ∉ terminalStates
                    sumOfDiscountedFutureRewards[a] = γ*sum(get(T, (s, a, sp), 0)*Up[sp] for sp in reachableStates(s))
                end

            end

            Up[s], π[s] = findmax(immediateReward[s, :] + sumOfDiscountedFutureRewards)

        end

        bellmanResidual = maximum(abs.(Up - U))

        @show k
        @show bellmanResidual

        k += 1
        U = deepcopy(Up)

    end

    return Up, collect(π)

end
