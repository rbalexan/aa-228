using CSV
using DataFrames

file = ARGS[1]

# load the dataset
dataset = CSV.read("data/" * file * ".csv")

# write the description
CSV.write("data/" * file * ".description", describe(dataset)[:, [:variable, :mean, :min, :median, :max, :eltype]])
