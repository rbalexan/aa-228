# using CSV
# using DataFrames
# using Printf
using Plots
using Distributions
using Random

# Define the problem structure
struct MultiFareDynamicPricingProblem
   timeHorizon::Int  # Time horizon
   totalTickets::Int # Number of seats in the plane
   ϵ::Real           # ϵ-greedy parameter
   η::Real           # learning rate in MDP update step
   γ::Real           # discount factor in MDP update step
   λ::Real           # eligibility trace parameter in MDP update step
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

include("spaceAndActionFunctions.jl")
include("generativeModel.jl")
include("chooseAction.jl")
include("solveMDP.jl")
include("getPolicy.jl")
include("runEpisodes.jl")

#Random.seed!(1) # for repeatability

# Specify fare classes
fareClasses = Dict(
    :business => FareClass(-1, 30, 700, 100, 20, 20.1, collect(range(550, 850, length=5))),
    #:leisure  => FareClass(-1, 20, 300,  50, 1,  10, collect(360:20:560)),
    #:mixed    => FareClass(2,  5, 400,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
problem  = MultiFareDynamicPricingProblem(20, 300, 0.2, 0.1, 1, 0.75, fareClasses)
solver   = :sarsa
episodes = 250000

Q, r = runEpisodes(problem, solver, episodes)

# Run model to get policy
jointPolicy, U, ticketsAvailableSpaceSize = getPolicy(problem, Q)

plot(1:100:episodes, r[1:100:end])
heatmap(Q, c=:viridis)

filename = "singleAgentSarsaLR01-250000"
heatmap(reshape(U, (ticketsAvailableSpaceSize, :)), framestyle=:box, dpi=600)
xlabel!("Time"); ylabel!("Tickets Available")
savefig("code/plots/" * filename * "-U.png")

heatmap(reshape(60*jointPolicy.+490, (ticketsAvailableSpaceSize, :)), clims=(490, 850), framestyle=:box, dpi=600, c=:Blues)
xlabel!("Time"); ylabel!("Tickets Available")
savefig("code/plots/" * filename * "-π.png")
