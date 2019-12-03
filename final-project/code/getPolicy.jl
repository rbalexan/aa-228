function getPolicy(p::Problem, Q::Array)

    # Initialize variables
    S_size = p.V*p.T
    π⋆ = zeros(Int, S_size)

    # Extract policy
    for s in 1:S_size
        π⋆[s] = argmax(Q[s,:])
    end

    # Return policy
    return π⋆[s]
end
