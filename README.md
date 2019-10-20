# aa-228

my workspace for AA 228: decision making under uncertainty

projects: [project 1](https://github.com/rbalexan/aa-228/tree/master/project-1) | [project 2](https://github.com/rbalexan/aa-228/tree/master/project-2) | [final project](https://github.com/rbalexan/aa-228/tree/master/final-project) 

### [project 1 | Bayesian structure learning](https://github.com/rbalexan/aa-228/tree/master/project-1/):

> A maximum likelihood approach to Bayesian structure learning assuming a uniform graph prior and a uniform Dirichlet network parameter prior (Bayesian-Dirichlet (BD) scoring). For a given graph structure, a dataset is read, counted, and scored using the Bayesian scoring function. Algorithms for maximizing the Bayesian score using directed graph search have been implemented.

- Bayesian score ([`bayesianCounts.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/bayesianCounts.jl) and [`bayesianScore.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/bayesianScore.jl))
- K2 directed graph search ([`singleK2Search.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/singleK2Search.jl)) with random restarts ([`multiK2Search.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/multiK2Search.jl)) 
- additional documentation ([`readme.pdf`](https://github.com/rbalexan/aa-228/tree/master/project-1/doc/README.pdf))

### [project 2 | reinforcement learning](https://github.com/rbalexan/aa-228/tree/master/project-2/):

>An implementation of dynamic programming for Markov decision processes (MDPs) using value iteration and policy iteration.

### [final project | dynamic pricing in the airline industry using reinforcement learning](https://github.com/rbalexan/aa-228/tree/master/final-project):

> Dynamic pricing in the airline industry demonstrates some of the most effective pricing schemes in business to maximize revenue based on customers' willingness to pay for particular goods at particular times. We propose to develop a dynamic pricing reinforcement learning algorithm to maximize revenue for a single flight with multiple customer segments. We suggest reinforcement learning as it is a model-free paradigm and thus less sensitive to unusual demand patterns, and because it is a relatively new approach to dynamic pricing for airlines.

