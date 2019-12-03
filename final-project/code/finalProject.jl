using CSV
using DataFrames
using Plots
using Printf
using Distributions

# Define the problem structure
struct Problem
   T::Int      # Time horizon
   V::Int      # Number of seats in the plane
   ϵ::Real     # ϵ-greedy parameter
   η::Real     # learning rate in MDP update step
   γ::Real     # decay parameter in MDP update step
   F::Dict     # Fare class parameters
end

struct FareClass

end

# Specify fare class parameters: α, β, w_μ, w_σ, k1, k2, A
# α, β: # customers; w_μ, w_σ: WTP threshold; k1, k2: WTP flexibility; A: action space
fare_class_parameters = Dict(
    :business => FareClass(1, 10, 800, 100, 1, 500, collect(760:10:850)),
    :leisure  => (5,  2, 400,  50, 1,  10, collect(360:10:450)),
    :mixed    => (2,  5, 500,  50, 1,  10, collect(460:10:550))
)

# Initialize the problem and global list of customers
p = Problem(10, 100, 0.2, 0.9, 0.9, fare_class_parameters)

customersWithTickets    = [Set() for i in 1:length(p.F)]
customersSeekingTickets = [Set() for i in 1:length(p.F)] # C

# Run generativeModel
generativeModel(p, 1, 20, 0, 18, 900)

# Get input and run
# if length(ARGS) != 1
#     error("usage: julia finalProject.jl <input>")
# end

# input = ARGS[1]

# compute(input)
