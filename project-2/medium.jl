using CSV
using DataFrames
using Plots
using Printf

include("inferTransitionAndReward.jl")
include("valueIteration.jl")
include("gaussSeidelValueIteration.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/medium.csv")

# compute state space and action space
𝒮  = collect(1:50000)
𝒜  = collect(1:7)
𝖲  = size(𝒮)[1]
𝖠  = size(𝒜)[1]

T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)

γ = 0.99
terminalStates  = [32464 29962 31463 30963 27963 28961 30464 33964 31965 29964 32965 29463 27462 30461 31461 28963]
reachableStates = (s) -> filter(x -> (x >= 1 && x <= 50000),
                        collect(s + i + 500j for i in -14:14, j in -3:3))
ϵ = 1000

U, π = valueIteration(           𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ)
U, π = gaussSeidelValueIteration(𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ)

writePolicy(π, "medium")

U = rotl90(reshape(U, (500, 100)))
π = rotl90(reshape(π, (500, 100)))

heatmap(reverse(U, dims=1), c=:viridis)
heatmap(reverse(π, dims=1), c=:viridis)
