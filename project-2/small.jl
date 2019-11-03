using CSV
using DataFrames
using Plots
using Printf

include("inferTransitionAndReward.jl")
include("valueIteration.jl")

# load the dataset
dataset = CSV.read("data/small.csv")

# compute state space and action space
𝒮  = collect(1:100)
𝒜  = collect(1:4)
𝖲  = size(𝒮)[1]
𝖠  = size(𝒜)[1]

T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)

γ = 0.95
terminalStates  = [15, 82]
reachableStates = (s) -> filter(x -> (x >= 1 && x <= 100), [s, s+1, s-1, s+10, s-10])
ϵ = 0.01

U, π = valueIteration(           𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ)