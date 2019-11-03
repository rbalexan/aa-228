function inferTransitionAndReward(dataset::DataFrame, 𝖲::Int, 𝖠::Int)

    # infer T and R from dataset

	# construct state-action counts, transition counts, and sums of rewards
	N  = Dict()
    Np = Dict()
    ρ  = Dict()

	# iterate over all samples and increment
    for i in 1:size(dataset)[1]

        s  = dataset.s[i]
        a  = dataset.a[i]
        r  = dataset.r[i]
        sp = dataset.sp[i]

		if r ≠ 0
			ρ[(s, a)]  = get(ρ,  (s, a),     0) + r
		end

        N[(s, a)]      = get(N,  (s, a),     0) + 1
        Np[(s, a, sp)] = get(Np, (s, a, sp), 0) + 1

    end

	# construct transition model and reward function
	T = Dict()
	R = Dict()

	# normalize transition counts and sums of rewards
	for key in collect(keys(Np))
		T[key] = Np[key] / N[key[1:2]]
	end

	for key in collect(keys(ρ))
		R[key] = ρ[key] / N[key]
	end

    return T, R

end
