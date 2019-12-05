function getPolicy(p::multiFareDynamicPricingProblem, iterations::Int)

   stateSpace = LinearIndices((0:p.totalTickets,1:p.timeHorizon))
   actionSpace = LinearIndices(zeros([length(p.fareClasses[f].fareActionSpace) for f in keys(p.fareClasses)]...))
   𝖲 = length(stateSpace)
   𝖠 = length(actionSpace)
   N = zeros(𝖲, 𝖠)
   Q = zeros(𝖲, 𝖠)

   for i = 1:iterations
      Q′ = solveMDP(p, stateSpace, actionSpace, 𝖲, 𝖠)
      N += [Q′[x]>0 for x in CartesianIndices(Q′)]
      Q += Q′
      @show "ITERATION======================================================================", iterations
      @show "Q", sum(Q)
   end

   averageQ = Q./(N+[N[x]==0 for x in CartesianIndices(N)])

      @show "averageQ======================================================================="
      @show "N", sum(averageQ)

   # Extract policy
   policy = argmax(Q, dims=2) # can replace with argmax(Q, dim=1), I think
   policy = [policy[s][2] for s in 1:𝖲]

   return policy
end
