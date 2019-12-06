function getPolicy(p::MultiFareDynamicPricingProblem, Q::Array)

   # Initialize state space
   stateSpace, 𝖲  = stateSpaceAttributes(p)
   actionSpace, _  = actionSpaceAttributes(p)

   # Extract policy
   policyIndices = argmax(Q, dims=2)

   U             = Q[policyIndices]
   jointPolicy   = [policyIndices[s][2] for s in 1:𝖲]
   #! agentPolicy[f] = ...
   # maybe access each agent's policy

   return jointPolicy, U

end
