function bayesianScore(n, q, r, m)

    score = 0 # uniform graph prior

    for i in 1:n

        for j in 1:q[i]

            αij0 = r[i] # for uniform Dirichlet prior
            mij0 = 0

            for k in 1:r[i]

                # αij0 = sum all αijk
                # mij0 = sum all mijk
                mij0 = mij0 + get(m, (i, j, k), 0)

            end

            for k in 1:r[i]
                score += lgamma(1 + get(m, (i, j, k), 0)) # for uniform Dirichlet prior
            end

            score += lgamma(αij0) - lgamma(αij0 + mij0)

        end

    end

    return score

end
