using CSV
using DataFrames

file   = ARGS[1]
policy = ARGS[2]

# write the policy
CSV.write("policies/" * file * ".policy", policy)
