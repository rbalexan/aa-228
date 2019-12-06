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

stateSpaceAttributes( p::MultiFareDynamicPricingProblem) = [(x, length(x)) for x in [LinearIndices((0:p.totalTickets,1:p.timeHorizon))]][1]
actionSpaceAttributes(p::MultiFareDynamicPricingProblem) = [(x, length(x)) for x in [LinearIndices(zeros([length(p.fareClasses[f].fareActionSpace) for f in keys(p.fareClasses)]...))]][1]

include("generativeModel.jl")
include("chooseAction.jl")
include("solveMDP.jl")
include("getPolicy.jl")

#Random.seed!(1) # for repeatability

# Solve MMDP for some number of episodes
function runEpisodes(p::MultiFareDynamicPricingProblem, solver::Symbol, episodes::Int)

   _, 𝖲  = stateSpaceAttributes(p)
   _, 𝖠  = actionSpaceAttributes(p)

   Q = zeros(𝖲, 𝖠)
   r = zeros(episodes)

   for episode in 1:episodes

      #Random.seed!(1) # for repeatability

      N = zeros(𝖲, 𝖠)
      Q, r[episode] = solveMDP(problem, solver, Q, N)
      #@show "EPISODE======================================================================", episode
      #@show "Q", sum(Q)
      if mod(episode, 100) == 0
         @show episode, mean(r[(episode-99):episode])
      end

   end

   return Q, r

end

# Specify fare classes
fareClasses = Dict(
    :business => FareClass(-1, 30, 700, 100, 20, 20.1, collect(range(550, 850, length=10))),
    #:leisure  => FareClass(-1, 20, 300,  50, 1,  10, collect(360:20:560)),
    #:mixed    => FareClass(2,  5, 400,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
problem  = MultiFareDynamicPricingProblem(20, 300, 0.2, 0.1, 1, 0.75, fareClasses)
solver   = :sarsaLambda
episodes = 25000

Q, r = runEpisodes(problem, solver, episodes)

# Run model to get policy
jointPolicy, U = getPolicy(problem, Q)

plot(1:episodes, r)
#plot()
heatmap(Q, c=:viridis)
heatmap(reshape(U, (301, :)))
heatmap(reshape(jointPolicy, (301, :)))
