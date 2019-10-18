function bayesianScore(n, q, r, m::Dict)

    # compute the Bayesian score of a graph given:
    #   1) a uniform graph prior
    #   2) a uniform Dirichlet parameter prior

    score = 0

    for i in 1:n, j in 1:q[i]

        αij0 = r[i]
        mij0 = 0

        for k in 1:r[i]
            mijk = get(m, (i, j, k), 0)
            mij0 += mijk
            score += lgamma(1 + mijk)
        end

        score += lgamma(αij0) - lgamma(αij0 + mij0)

    end

    return score

end
