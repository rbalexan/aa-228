using CSV
using DataFrames
using Plots
using Printf

# load the dataset
dataset = CSV.read("data/small.csv")

# compute state space and action space
𝒮  = collect(1:100)
𝒜  = collect(1:4)
𝖲  = size(𝒮)[1]
𝖠  = size(𝒜)[1]

