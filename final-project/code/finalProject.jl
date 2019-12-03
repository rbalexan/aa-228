# using CSV
# using DataFrames
# using Plots
# using Printf
using Distributions
using Random

# Define the problem structure
struct MultiFareDynamicPricingProblem
   T::Int      # Time horizon
   V::Int      # Number of seats in the plane
   ϵ::Real     # ϵ-greedy parameter
   η::Real     # learning rate in MDP update step
   γ::Real     # decay parameter in MDP update step
   F::Dict     # Fare class parameters
end

# Define the fare class stucture
struct FareClass
   customerArrivalSlope::Real          # α
   customerArrivalIntercept::Real      # β
   wtpThresholdMean::Real              # w_μ
   wtpThresholdStandardDeviation::Real # w_σ
   wtpFlexibilityLowerBound::Real      # k1
   wtpFlexibilityUpperBound::Real      # k2
   actionSpace::Vector
end

# Define the customer structure
struct Customer
     wtpThreshold::Real   # w
     wtpFlexibility::Real # k
end

include("generativeModel.jl")
Random.seed!(1) # for repeatability

# Specify fare classes
fareClasses = Dict(
    :business => FareClass(1, 10, 800, 100, 1, 500, collect(760:10:850)),
    :leisure  => FareClass(5,  2, 400,  50, 1,  10, collect(360:10:450)),
    :mixed    => FareClass(2,  5, 500,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
problem = MultiFareDynamicPricingProblem(10, 100, 0.2, 0.9, 0.9, fareClasses)

customersWithoutTickets = Dict(k => Set() for k in keys(fareClasses)) # C
customersWithTickets    = Dict(k => Set() for k in keys(fareClasses))

# Run generativeModel
ticketsAvailable, ticketsSold, fareClassReward = generativeModel(problem, :business, 20, 18, 900)

#for t in 1:30, fareClass in keys(fareClasses)
#   ticketsAvailable, ticketsSold, fareClassReward = generativeModel(problem, fareClass, 20, t, policy[fareClass][s])
#end

# Get input and run
# if length(ARGS) != 1
#     error("usage: julia finalProject.jl <input>")
# end

# input = ARGS[1]

# compute(input)
