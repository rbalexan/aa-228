function modalPolicyUpdating(𝖲, U, π, N)

    πp = deepcopy(π)

    for s in 1:𝖲

        if U[s] == 0

            # return the neighbors of s where we have a nonzero U value
            neighborhood   = N(s, 1)
            validNeighbors = findall(U[neighborhood] .!= 0)

            # compute the policy mode for the valid neighbors
            if !isempty(validNeighbors)
                πp[s] = max(0, mode(π[neighborhood[validNeighbors]]))
            end

        end

    end

    return πp

end
