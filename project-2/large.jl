using CSV
using DataFrames
using Plots
using Printf

include("inferTransitionAndReward.jl")
include("valueIteration.jl")
include("gaussSeidelValueIteration.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/large.csv")

# compute state space and action space
𝒮  = collect(1:320000) #312020
𝒜  = collect(1:9)
𝖲  = size(𝒮)[1]
𝖠  = size(𝒜)[1]

T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)

γ = 0.95
terminalStates = []#151313, 151202]
reachableStateSpace = unique(dataset.s)
reachableStates = (s) -> reachableStateSpace
ϵ = 0.1

U, π = valueIteration(           𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ)
U, π = gaussSeidelValueIteration(𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ, reachableStateSpace)

writePolicy(π[1:312020], "large")

gr()
U = rotl90(reshape(U, (10000, 32)))
heatmap(reverse(U[32 .- [15, 23, 27, 29, 30], sort(unique(mod.(uniqueStates, 10000)))], dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("plots/large_U.png")
π = rotl90(reshape(π, (10000, 32)))
heatmap(reverse(π[32 .- [15, 23, 27, 29, 30], sort(unique(mod.(uniqueStates, 10000)))], dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("help plots/large_π.png")
