using CSV
using DataFrames
using Plots
using Printf
using LinearAlgebra
using Distributions

include("inferTransitionAndReward.jl")
include("modalPolicyFilling.jl")
include("plotMedium.jl")
include("sarsaLambdaLearning.jl")
include("sarsaLambdaLearningProportionate.jl")
include("sarsaLambdaLearningLocalApproximation.jl")
include("sarsaLambdaLearningGlobalApproximation.jl")
include("valueIteration.jl")
include("valueIterationGaussSeidel.jl")
include("writeParameters.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/medium.csv")

# set size of state space and action space
𝖲 = 50000
𝖠 = 7

# define reachable states of s
reachableStates = (s) -> filter(x -> (x >= 1 && x <= 50000),
                        collect(s + i + 500j for i in -15:15, j in -3:3))

# solve function
function solve(solutionType, 𝖲, 𝖠, dataset, reachableStates, α, β, γ, ϵ, λ, N)

    if     solutionType == "VI"
        U, π = valueIteration(                        𝖲, 𝖠, dataset, reachableStates, γ, ϵ)
    elseif solutionType == "GSVI"
        U, π = valueIterationGaussSeidel(             𝖲, 𝖠, dataset, reachableStates, γ, ϵ)
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
γ = 0.99
ϵ = 1000.0
α = 0.05
λ = 0.9

# define neighborhood of s
N = (s, a) -> filter(x -> (x >= 1 && x <= 50000 && x ≠ s &&
                        sqrt(((mod(x, 500) - mod(s, 500))/5)^2 + (ceil(x/500) - ceil(s/500))^2) <= 1.5),
                        collect(s + i + 500j for i in -10:10, j in -2:2))

# define the global approximation function (fourier series decomposition)
f = 3
β = (s, a) -> vcat(reshape([a==k for k in 1:𝖠], :, 1),
    reshape([(a==k)*cos(i*2pi/10*mod(s-1, 10))*cos(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*cos(i*2pi/10*mod(s-1, 10))*sin(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*sin(i*2pi/10*mod(s-1, 10))*cos(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1),
    reshape([(a==k)*sin(i*2pi/10*mod(s-1, 10))*sin(j*2pi/10*floor(s/10)) for i in 1:f, j in 1:f, k in 1:𝖠], :, 1))

# solve
(U, π), t = @timed solve(solutionType, 𝖲, 𝖠, dataset, reachableStates, α, β, γ, ϵ, λ, N)

@show t

# compute the modally-filled policy
πModal = modalPolicyFilling(𝖲, U, π, N)

# output policy and parameter files and plot the solutions
writePolicy(π,      "medium_" * solutionType)
writePolicy(πModal, "medium_" * solutionType * "_modal")

writeParameters(γ, ϵ, α, λ, f, t, "medium_" * solutionType)

plotMedium(U, π, πModal, solutionType)
