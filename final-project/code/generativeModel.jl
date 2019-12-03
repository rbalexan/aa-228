function generativeModel(p::Problem, f::Int, v::Int, u::Int, t::Int, a::Int)

    struct Customer
        # i::Int  # index
        wtpThreshold::Real   # w
        wtpFlexibility::Real # k
    end

    # Initialize variables
    ticketsSold = 0                                          # u'
    customersWithTickets    = [Set() for i in 1:length(p.F)]
    customersSeekingTickets = [Set() for i in 1:length(p.F)] # C

    α, β, w_μ, w_σ, k1, k2 = p.F[f]
    customerArrivalDistribution = Poisson(α*(p.T-t) + β) # n-distribution
    WTPThresholdDistribution    = Normal(μ, σ)           # w-distribution
    WTPFlexibilityDistribution  = Uniform(k1, k2)        # k-distribution

    # Sample the number of new customers from the Poisson distribution
    newCustomers = rand(customerArrivalDistribution, 1)[] # n
    #@show n

    # Add new customers for this time step
    for newCustomer in 1:newCustomers

        # for existingCustomer in customersSeekingTickets
        #     push!(existingCustomerIndices, existingCustomer.i)
        # end

        # get first available index, WTP parameters
        # i = first(symdiff(1:10000, existingCustomerIndices)
        wtpThreshold   = rand(wtpThresholdDistribution,   1)[] # w
        wtpFlexibility = rand(wtpFlexibilityDistribution, 1)[] # k

        customer = Customer(wtpThreshold, wtpFlexibility)
        push!(customersSeekingTickets[f], customer)
        #@show "New customer", i, w, k, length(C[f])

    end

    # Check to see which customers will purchase tickets
    for customer in deepcopy(customersSeekingTickets[f])

        # compute the purchase probability and sample from its Bernoulli distribution
        purchaseProbability  = ccdf(Logistic(customer.w, customer.k), a) # ϕ value
        purchaseDistribution = Bernoulli(purchaseProbability)
        purchase             = rand(purchaseDistribution, 1)[]

        if purchase

            ticketsSold += 1
            pop!( customersSeekingTickets[f], customer)
            push!(customersWithTickets[f],    customer)

        end
        #@show "Check", [round(x,digits=0) for x in j], round(ϕ*100,digits=0), b

    end

    # Calculate current reward
    reward = a * ticketsSold

    # Return results
    return (v-u′, u′, t+1, reward)

end
