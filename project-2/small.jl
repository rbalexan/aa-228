using CSV
using DataFrames
using Plots
using Printf
using LinearAlgebra
using Distributions

include("inferTransitionAndReward.jl")
include("modalPolicyFilling.jl")
include("plotSmall.jl")
include("sarsaLambdaLearning.jl")
include("sarsaLambdaLearningProportionate.jl")
include("sarsaLambdaLearningLocalApproximation.jl")
include("sarsaLambdaLearningGlobalApproximation.jl")
include("valueIteration.jl")
include("valueIterationGaussSeidel.jl")
include("writeParameters.jl")
include("writePolicy.jl")

# load the dataset
dataset = CSV.read("data/small.csv")

# set size of state space and action space
𝖲 = 100
𝖠 = 4

# define reachable states of s
reachableStates = (s) -> filter(x -> (x >= 1 && x <= 100), [s, s+1, s-1, s+10, s-10])

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
γ = 0.95
ϵ = 0.01
α = 0.05
λ = 0.9

# define neighborhood of s
N = (s, a) -> filter(x -> (x >= 1 && x <= 100 && x ≠ s &&
                    sqrt((mod(x, 10) - mod(s, 10))^2 + (ceil(x/10) - ceil(s/10))^2) <= 1.5),
                    collect(s + i + 10j for i in -2:2, j in -2:2))

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
writePolicy(π,      "small_" * solutionType)
writePolicy(πModal, "small_" * solutionType * "_modal")

writeParameters(γ, ϵ, α, λ, f, t, "small_" * solutionType)

plotSmall(U, π, πModal, solutionType)
