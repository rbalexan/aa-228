using CSV
using DataFrames
using Plots
using Printf
using Distributions

# Define the problem structure
struct Problem
   T::Int # Time horizon
   V::Int # Number of seats in the plane
   ϵ::Float64 # ϵ-greedy parameter
   η::Float64 # learning rate in MDP update step
   γ::Float # decay parameter in MDP update step
   F::Dict # Fare class parameters
end

# Specify fare class parameters: α, β, w_μ, w_σ, k1, k2, A
# α, β: # customers; w_μ, w_σ: WTP threshold; k1, k2: WTP flexibility; A: action space
fare_class_parameters = Dict(
    1 => (1,10,800,100,1,500,[x for x in 760:10:850]),
    2 => (5,2,400,50,1,10,[x for x in 360:10:450]),
    3 => (2,5,500,50,1,10,[x for x in 460:10:550])
)

# Initialize the problem and global list of customers
p = Problem(10, 100, 0.2, 0.9, 0.9, fare_class_parameters)
C = [[] for i=1:length(p.F)]

# Run generativeModel
generativeModel(p, 1, 20, 0, 18, 900)

# Get input and run
# if length(ARGS) != 1
#     error("usage: julia finalProject.jl <input>")
# end

# input = ARGS[1]

# compute(input)
