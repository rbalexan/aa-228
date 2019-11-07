function sarsaLambdaLearningGlobalApproximation(𝖲::Int, 𝖠::Int, dataset::DataFrame, β,
    α::Float64=0.9, γ::Float64=1.0, λ::Float64=0.9)

    b = size(β(1,2))[1]
    # initialize θ and N
    θ = zeros(b)
    N = zeros(𝖲, 𝖠)
    Q = zeros(𝖲, 𝖠)
    A = zeros(𝖲*𝖠, b)
    B = zeros(𝖲, 𝖠, b)

    # Hankel matrix (for second-order polynomial) and its singular value decomposition
    for i in 1:𝖲*𝖠
        A[i, :] = β(mod(i,𝖲), ceil(i/𝖲))
    end

    for s in 1:𝖲, a in 1:𝖠
        B[s, a, :] = β(s, a)
    end


    F = svd(A)
    W = F.V*inv(Diagonal(F.S))*F.U'

    # loop over the dataset
    for i in 1:size(dataset)[1]

        # if we are at the end of an episode, reset the counts for next episode
        # and skip the last sarsa iteration
        if i == size(dataset)[1] || dataset.sp[i] ≠ dataset.s[i+1]
            N = zeros(𝖲, 𝖠)
            continue
        end

        s   = dataset.s[i]
        a   = dataset.a[i]
        r   = dataset.r[i]
        sp  = dataset.s[i+1]
        ap  = dataset.a[i+1]

        N[s, a] += 1
        δ       =  r + γ*sum(θ.*B[sp, ap, :]) - sum(θ.*B[s, a, :])

        for s in 1:𝖲, a in 1:𝖠

            Q[s, a] += α*δ*N[s, a]
            N[s, a] *= γ*λ

        end

        # predicted model
        θ = W*reshape(Q, 400, :)

    end

    Q = zeros(𝖲, 𝖠)
    for s in 1:𝖲, a in 1:𝖠
        Q[s, a] = sum(θ.*B[s, a, :])
    end

    πInd = argmax(Q, dims=2)

    U = Q[πInd]
    π = collect(πInd[i][2] for i in 1:𝖲)

    return U, π

end
