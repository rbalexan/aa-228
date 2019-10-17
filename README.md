# aa-228

my workspace for AA 228: decision making under uncertainty

projects: [project 0](https://github.com/rbalexan/aa-228/tree/master/project-0) | [project 1](https://github.com/rbalexan/aa-228/tree/master/project-1) | [project 2](https://github.com/rbalexan/aa-228/tree/master/project-2) | [final project](https://github.com/rbalexan/aa-228/tree/master/final-project) 

### [project 0](https://github.com/rbalexan/aa-228/tree/master/project-0):

sample submission file

###[project 1](https://github.com/rbalexan/aa-228/tree/master/project-1/project1.jl):

> A maximum likelihood approach to Bayesian structure learning assuming a uniform graph prior and a uniform Dirichlet network parameter prior (Bayesian Dirichlet equivalent uniform (BDeu) scoring). For a given graph structure, a dataset is read, counted, and scored using the Bayesian scoring function. Algorithms for maximizing the Bayesian score using graph search have been implemented, which include both directed and partially-directed graph search.

- Bayesian score implementation ([`bayesianScore.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/bayesianScore.jl))
- directed graph search algorithms:
  - K2 search
  - local search (with randomized restart through [`initializeRandomGraph.jl`](https://github.com/rbalexan/aa-228/tree/master/project-1/initializeRandomGraph.jl))
- partially directed graph search algorithms:
  - local search (with randomized restart)
- additional documentation: [readme.pdf](https://github.com/rbalexan/aa-228/tree/master/project-1/doc/README.pdf)

### [final project](https://github.com/rbalexan/aa-228/tree/master/final-project):

proposal:

> Dynamic pricing in the airline industry demonstrates some of the most effective pricing schemes in business to maximize revenue based on customers' willingness to pay for particular goods at particular times. We propose to develop a dynamic pricing reinforcement learning algorithm to maximize revenue for a single flight with multiple customer segments. We suggest reinforcement learning as it is a model-free paradigm and thus less sensitive to unusual demand patterns, and because it is a relatively new approach to dynamic pricing for airlines.

