function valueIteration(𝖲::Int, 𝖠::Int, T::Dict, R::Dict, γ::Float64,
    terminalStates, reachableStates, ϵ=1)

    δ = ϵ*(1-γ)/γ
    bellmanResidual = δ+1
    @show δ

    U  = zeros(𝖲)
    Up = zeros(𝖲)
    π  = zeros(𝖲)

    sumOfDiscountedFutureRewards = zeros(𝖲, 𝖠)
    immediateReward              = zeros(𝖲, 𝖠)

    for s in 1:𝖲, a in 1:𝖠
        immediateReward[s, a] = get(R, (s, a), 0)
    end

    k = 1

    while bellmanResidual > δ

        sumOfDiscountedFutureRewards = zeros(𝖲, 𝖠)

        for s in filter(x -> x ∉ terminalStates, 1:𝖲), a in 1:𝖠 # could convert to reachable state space

            sumOfDiscountedFutureRewards[s, a] = γ*sum(get(T, (s, a, sp), 0)*Up[sp] for sp in reachableStates(s))

        end

        Up, π = findmax(immediateReward + sumOfDiscountedFutureRewards, dims=2)
        bellmanResidual = maximum(abs.(Up - U))

        @show k
        @show bellmanResidual

        k += 1
        U = deepcopy(Up)

    end

    return Up, collect(π[i][2] for i in 1:𝖲)

end
