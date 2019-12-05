function getPolicy(p::multiFareDynamicPricingProblem, iterations::Int)

   stateSpace  = LinearIndices((0:p.totalTickets,1:p.timeHorizon))
   actionSpace = LinearIndices(zeros([length(p.fareClasses[f].fareActionSpace) for f in keys(p.fareClasses)]...))
   𝖲 = length(stateSpace)
   𝖠 = length(actionSpace)
   Q = zeros(𝖲, 𝖠)

   for i in 1:iterations
# need to pass Q into solve MDP for iteration
      Q, r = solveMDP(p, stateSpace, actionSpace, 𝖲, 𝖠, deepcopy(Q))
      @show "ITERATION======================================================================", iterations
      @show "Q", sum(Q)
      @show r
   end

   # Extract policy
   policyIndices = argmax(Q, dims=2)

   U             = Q[policyIndices]
   jointPolicy   = [policyIndices[s][2] for s in 1:𝖲]
   #agentPolicy[f] = ...
# maybe access each agent's policy

   return jointPolicy, U
end
