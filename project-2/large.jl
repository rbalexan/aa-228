using CSV
using DataFrames
using Plots
using Printf
using LinearAlgebra
using Distributions

include("inferTransitionAndReward.jl")
include("modalPolicyFilling.jl")
include("plotLarge.jl")
include("sarsaLambdaLearning.jl")
include("sarsaLambdaLearningProportionate.jl")
include("sarsaLambdaLearningLocalApproximation.jl")
include("sarsaLambdaLearningGlobalApproximation.jl")
include("valueIteration.jl")
include("valueIterationGaussSeidel.jl")
include("writeParameters.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/large.csv")

# set size of state space and action space
𝖲 = 320000 # 312020
𝖠 = 9

# define reachable states of s
reachableStateSpace = unique(dataset.s)
reachableStates     = (s) -> reachableStateSpace

# solve function
function solve(solutionType, 𝖲, 𝖠, dataset, reachableStates, α, β, γ, ϵ, λ, N, reachableStateSpace)

    if     solutionType == "VI"
        U, π = valueIteration(                        𝖲, 𝖠, dataset, reachableStates, γ, ϵ)
    elseif solutionType == "GSVI"
        U, π = valueIterationGaussSeidel(             𝖲, 𝖠, dataset, reachableStates, γ, ϵ, reachableStateSpace)
    elseif solutionType == "sarsaLambda"
        U, π = sarsaLambdaLearning(                   𝖲, 𝖠, dataset, α, γ, λ)
    elseif solutionType == "sarsaLambdaProp"
        U, π = sarsaLambdaLearningProportionate(      𝖲, 𝖠, dataset, γ, λ)
    elseif solutionType == "sarsaLambdaLA"
        U, π = sarsaLambdaLearningLocalApproximation( 𝖲, 𝖠, dataset, N, α, γ, λ)
    elseif solutionType == "sarsaLambdaGA"
        U, π = sarsaLambdaLearningGlobalApproximation(𝖲, 𝖠, dataset, β, α, γ, λ)
    end

    return U, π

end

# solution type
solutionType = "GSVI"

# solution parameters
γ = 0.95
ϵ = 0.1
α = 0.1
λ = 0.9

# define neighborhood of s
N = (s, a) -> 0

# define the global approximation function (fourier series decomposition)
β = (s, a) -> 0

# solve
(U, p), t = @timed solve(solutionType, 𝖲, 𝖠, dataset, reachableStates, α, β, γ, ϵ, λ, N, reachableStateSpace)

@show t

# output policy and parameter files and plot the solutions
writePolicy(p[1:312020], "large_" * solutionType)

writeParameters(γ, ϵ, α, λ, 0, t, "large_" * solutionType)

plotLarge(U, p, solutionType, reachableStateSpace)
