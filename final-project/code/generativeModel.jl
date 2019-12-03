function generativeModel(p::Problem, f::Int, v::Int, u::Int, t::Int, a::Int)

    # Initialize variables
    u′ = 0
    paying_customers = []
    α, β, w_μ, w_σ, k1, k2 = p.F[f]
    n_dist = Poisson(α*(p.T-t) + β)
    w_dist = Normal(w_μ, w_σ)
    k_dist = Uniform(k1, k2)
    n = rand(n_dist,1)[]
    #@show n

    # Add new cutomers for this time step
    for i = 1:n
        w = rand(w_dist,1)[]
        k = rand(k_dist,1)[]
        push!(C[f], [i,w,k])
        #@show "New customer", i, w, k, length(C[f])
    end

    # Check to see which customers will purchase tickets
    for j in C[f]
        i, w, k = j
        ϕ = 1 - cdf(Logistic(w,k),a)
        b_dist = Bernoulli(ϕ)
        b = rand(b_dist,1)[]
        if b == 1
            u′ += 1
            push!(paying_customers, j)
        end
        #@show "Check", [round(x,digits=0) for x in j], round(ϕ*100,digits=0), b
    end

    # Remove paying customers from global list of customers
    setdiff!(C[f], paying_customers)

    # Calculate current reward
    r = a * u′

    # Return results
    return (v-u′,u′,t+1,r)
end
