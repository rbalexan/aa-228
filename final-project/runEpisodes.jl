function runEpisodes(p::MultiFareDynamicPricingProblem, solver::Symbol, episodes::Int)
# Solve MMDP for some number of episodes

   # Initialize state and action spaces
   _, 𝖲  = stateSpaceAttributes(p)
   _, 𝖠  = actionSpaceAttributes(p)

   # Initialize Q and r
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
