# using CSV
# using DataFrames
# using Plots
# using Printf
using Distributions
using Random

# Define the problem structure
struct multiFareDynamicPricingProblem
   timeHorizon::Int  # Time horizon
   totalTickets::Int   # Number of seats in the plane
   ϵ::Real           # ϵ-greedy parameter
   η::Real           # learning rate in MDP update step
   γ::Real           # decay parameter in MDP update step
   fareClasses::Dict # Fare class parameters
end

# Define the fare class stucture
struct fareClass
   customerArrivalSlope::Real          # α
   customerArrivalIntercept::Real      # β
   wtpThresholdMean::Real              # w_μ
   wtpThresholdStandardDeviation::Real # w_σ
   wtpFlexibilityLowerBound::Real      # k1
   wtpFlexibilityUpperBound::Real      # k2
   actionSpace::Vector
end

# Define the customer structure
struct customer
     wtpThreshold::Real   # w
     wtpFlexibility::Real # k
end

include("generativeModel.jl")
include("solveMDP.jl")
#Random.seed!(1) # for repeatability

# Specify fare classes
fareClasses = Dict(
    :business => fareClass(1, 10, 800, 100, 1, 500, collect(760:10:850)),
    :leisure  => fareClass(5,  2, 400,  50, 1,  10, collect(360:10:450)),
    :mixed    => fareClass(2,  5, 500,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
problem = multiFareDynamicPricingProblem(20, 100, 0.2, 0.9, 0.9, fareClasses)

customersWithoutTickets = Dict(k => Set() for k in keys(fareClasses)) # C
customersWithTickets    = Dict(k => Set() for k in keys(fareClasses))

# Run generativeModel
# ticketsAvailable, ticketsSold, fareClassReward, customersWithPurchase = generativeModel(problem, :business, 20, 18, 900)
# @show ticketsAvailable, ticketsSold, fareClassReward, customersWithPurchase

solveMDP(problem)

#for t in 1:30, fareClass in keys(fareClasses)
#   ticketsAvailable, ticketsSold, fareClassReward = generativeModel(problem, fareClass, 20, t, policy[fareClass][s])
#end

# Get input and run
# if length(ARGS) != 1
#     error("usage: julia finalProject.jl <input>")
# end

# input = ARGS[1]

# compute(input)
