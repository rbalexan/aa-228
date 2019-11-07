using CSV
using DataFrames
using Plots
using Printf
using LinearAlgebra

include("inferTransitionAndReward.jl")
include("valueIteration.jl")
include("valueIterationGaussSeidel.jl")
include("sarsaLambdaLearning.jl")
include("sarsaLambdaLearningProportionate.jl")
include("sarsaLambdaLearningGlobalApproximation.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/small.csv")

# compute state space and action space
𝒮  = collect(1:100)
𝒜  = collect(1:4)
𝖲  = size(𝒮)[1]
𝖠  = size(𝒜)[1]

γ = 0.95
terminalStates  = [15, 82]
reachableStates = (s) -> filter(x -> (x >= 1 && x <= 100), [s, s+1, s-1, s+10, s-10])
ϵ = 0.01
α = 0.1
λ = 0.7

f = 4
β = (s, a) -> vcat(reshape([a==k for k in 1:𝖠], :, 1),
    reshape([(a==k)*cos(i*2pi/10*mod(s-1, 10))*cos(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*cos(i*2pi/10*mod(s-1, 10))*sin(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*sin(i*2pi/10*mod(s-1, 10))*cos(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*sin(i*2pi/10*mod(s-1, 10))*sin(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1))

U, π = valueIteration(           𝖲, 𝖠, dataset, γ, terminalStates, reachableStates, ϵ)
U, π = valueIterationGaussSeidel(𝖲, 𝖠, dataset, γ, terminalStates, reachableStates, ϵ)

U, π = sarsaLambdaLearning(                   𝖲, 𝖠, dataset, α, γ, λ)
U, π = sarsaLambdaLearningProportionate(      𝖲, 𝖠, dataset, γ, λ)
U, π = sarsaLambdaLearningGlobalApproximation(𝖲, 𝖠, dataset, β, α, γ, λ)

writePolicy(π, "small")

gr()
heatmap(reverse(rotl90(reshape(U, (10, 10))), dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("plots/small_U.png")
heatmap(reverse(rotl90(reshape(π, (10, 10))), dims=1), c=:viridis, framestyle=:box, dpi=600)
savefig("plots/small_π.png")


# left, right, up, down
