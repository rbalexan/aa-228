function sarsaLambdaLearningProportionate(𝖲::Int, 𝖠::Int, dataset::DataFrame,
    γ::Float64=1.0, λ::Float64=0.9)

    # initialize Q, N, and Nd
    Q  = zeros(𝖲, 𝖠)
    N  = zeros(𝖲, 𝖠)
    Nd = zeros(𝖲, 𝖠)

    # loop over the dataset
    for i in 1:size(dataset)[1]

        # if we are at the end of an episode, reset the counts for next episode
        # and skip the last sarsa iteration
        if i == size(dataset)[1] || dataset.sp[i] ≠ dataset.s[i+1]
            N  = zeros(𝖲, 𝖠)
            Nd = zeros(𝖲, 𝖠)
            continue
        end

        s   = dataset.s[i]
        a   = dataset.a[i]
        r   = dataset.r[i]
        sp  = dataset.s[i+1]
        ap  = dataset.a[i+1]

        N[s, a]  += 1 # regular
        Nd[s, a] += 1 # eligibility trace
        δ        =  r + γ*Q[sp, ap] - Q[s, a]

        #for s in 1:𝖲, a in 1:𝖠
        Q  += 1*δ./N.*Nd
        replace!(Q, NaN=>0)
        Nd *= γ*λ

    end

    πInd = argmax(Q, dims=2)

    U = Q[πInd]
    π = collect(πInd[i][2] for i in 1:𝖲)

    return U, π

end
