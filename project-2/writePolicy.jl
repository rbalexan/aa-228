function writePolicy(π::Array, file)

# write the policy
    open("policies/" * file * ".policy", "w") do io
        for i in 1:length(π)
            @printf(io, "%i\n", max(π[i], 1)) # write a default policy if we have 0
        end
    end

end
