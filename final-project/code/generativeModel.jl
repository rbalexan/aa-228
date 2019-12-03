function generativeModel(p::MultiFareDynamicPricingProblem, f::Symbol,
    ticketsAvailable::Int, t::Int, ticketPrice::Real)

    # Initialize variables
    ticketsSold = 0   # u

    fareClass = p.F[f]
    customerArrivalDistribution = Poisson(fareClass.customerArrivalSlope*(p.T - t) + fareClass.customerArrivalIntercept) # n-distribution
    wtpThresholdDistribution    = Normal( fareClass.wtpThresholdMean,         fareClass.wtpThresholdStandardDeviation)   # w-distribution
    wtpFlexibilityDistribution  = Uniform(fareClass.wtpFlexibilityLowerBound, fareClass.wtpFlexibilityUpperBound)        # k-distribution

    # Sample the number of new customers from the Poisson distribution
    newCustomers = rand(customerArrivalDistribution, 1)[] # n
    #@show n

    # Add new customers for this time step
    for newCustomer in 1:newCustomers

        # get WTP parameters
        wtpThreshold   = rand(wtpThresholdDistribution,   1)[] # w
        wtpFlexibility = rand(wtpFlexibilityDistribution, 1)[] # k

        customer = Customer(wtpThreshold, wtpFlexibility)
        push!(customersWithoutTickets[f], customer)
        #@show "New customer", i, w, k, length(C[f])

    end

    # Check to see which customers will purchase tickets
    for customer in deepcopy(customersWithoutTickets[f])

        # compute the purchase probability and sample from its Bernoulli distribution
        purchaseProbability  = ticketPrice <= customer.wtpThreshold ?
            1 : ccdf(Logistic(customer.wtpThreshold, customer.wtpFlexibility), ticketPrice) # ϕ value
        purchaseDistribution = Bernoulli(purchaseProbability)
        purchase             = Bool(rand(purchaseDistribution, 1)[])

        if purchase

            ticketsSold += 1
            pop!( customersWithoutTickets[f], customer)
            push!(customersWithTickets[f],    customer)

        end
        #@show "Check", [round(x,digits=0) for x in j], round(ϕ*100,digits=0), b

    end

    # Calculate current reward
    revenue = ticketPrice * ticketsSold

    # Return results
    return ticketsAvailable - ticketsSold, ticketsSold, revenue

end
