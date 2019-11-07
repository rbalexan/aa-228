using CSV
using DataFrames
using Plots
using Printf

include("inferTransitionAndReward.jl")
include("valueIteration.jl")
include("valueIterationGaussSeidel.jl")
include("writePolicy.jl")

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
U, π = valueIterationGaussSeidel(𝖲, 𝖠, T, R, γ, terminalStates, reachableStates, ϵ)

writePolicy(π, "small")

gr()
heatmap(reverse(rotl90(reshape(U, (10, 10))), dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("plots/small_U.png")
heatmap(reverse(rotl90(reshape(π, (10, 10))), dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("plots/small_π.png")


# left, right, up, down
