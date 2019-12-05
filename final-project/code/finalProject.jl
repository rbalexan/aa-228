# using CSV
# using DataFrames
# using Plots
# using Printf
using Distributions
using Random

# Define the problem structure
struct MultiFareDynamicPricingProblem
   timeHorizon::Int  # Time horizon
   totalTickets::Int # Number of seats in the plane
   ϵ::Real           # ϵ-greedy parameter
   η::Real           # learning rate in MDP update step
   γ::Real           # decay parameter in MDP update step
   fareClasses::Dict # Fare class parameters
end

# Define the fare class stucture
struct FareClass
   customerArrivalSlope::Real          # α
   customerArrivalIntercept::Real      # β
   wtpThresholdMean::Real              # w_μ
   wtpThresholdStandardDeviation::Real # w_σ
   wtpFlexibilityLowerBound::Real      # k1
   wtpFlexibilityUpperBound::Real      # k2
   fareActionSpace::Vector
end

# Define the customer structure
struct Customer
     wtpThreshold::Real   # w
     wtpFlexibility::Real # k
end

stateSpaceAttributes(p::MultiFareDynamicPricingProblem)  = [(x, length(x)) for x in [LinearIndices((0:p.totalTickets,1:p.timeHorizon))]][1]
actionSpaceAttributes(p::MultiFareDynamicPricingProblem) = [(x, length(x)) for x in [LinearIndices(zeros([length(p.fareClasses[f].fareActionSpace) for f in keys(p.fareClasses)]...))]][1]

include("generativeModel.jl")
include("chooseAction.jl")
include("solveMDP.jl")
include("getPolicy.jl")

#Random.seed!(1) # for repeatability

# Specify fare classes
fareClasses = Dict(
    :business => FareClass(1, 10, 700, 100, 1, 500, collect(760:10:850)),
    :leisure  => FareClass(5,  2, 300,  50, 1,  10, collect(360:10:450)),
    :mixed    => FareClass(2,  5, 400,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
problem = MultiFareDynamicPricingProblem(20, 1000, 0.2, 0.9, 1, fareClasses)

#! Solve MMDP for some number of episodes
# for i in 1:episodes
#  Q, r= solveMDP(asdnaskjasbdkja, Q)
# r T[i] = r
# end

# Run model to get policy
jointPolicy, U = getPolicy(problem, 2)
