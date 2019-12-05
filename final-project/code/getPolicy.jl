function getPolicy(p::MultiFareDynamicPricingProblem, iterations::Int)

   # Initialize state and action spaces
   stateSpace, 𝖲  = stateSpaceAttributes(p)
   actionSpace, 𝖠 = actionSpaceAttributes(p)
   Q = zeros(𝖲, 𝖠)

   for i in 1:iterations
      Q, r = solveMDP(p, deepcopy(Q))
      @show "ITERATION======================================================================", iterations
      @show "Q", sum(Q)
      @show r
      #! Keep track of rewards
   end

   # Extract policy
   policyIndices = argmax(Q, dims=2)

   U             = Q[policyIndices]
   jointPolicy   = [policyIndices[s][2] for s in 1:𝖲]
   #! agentPolicy[f] = ...
   # maybe access each agent's policy

   return jointPolicy, U
end
