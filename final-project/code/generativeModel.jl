function generativeModel(p::MultiFareDynamicPricingProblem, f::Symbol,
    ticketsAvailable::Int, t::Int, a::Dict, customersWithoutTickets::Dict)

    # Initialize variables
    ticketsSold = 0   # u
    customersWithoutTicketsByFareClass = customersWithoutTickets[f]

    fareClass = p.fareClasses[f]
    λ = max(0, fareClass.customerArrivalSlope*t + fareClass.customerArrivalIntercept)                  # Poisson parameter
    customerArrivalDistribution = Poisson(λ)                                                                            # n-distribution
    wtpThresholdDistribution    = Normal( fareClass.wtpThresholdMean,         fareClass.wtpThresholdStandardDeviation)  # w-distribution
    wtpFlexibilityDistribution  = Uniform(fareClass.wtpFlexibilityLowerBound, fareClass.wtpFlexibilityUpperBound)       # k-distribution

    # Sample the number of new customers from the Poisson distribution
    newCustomers = rand(customerArrivalDistribution) # n

    # Add new customers for this time step
    for newCustomer in 1:newCustomers

        # Get WTP parameters
        wtpThreshold   = rand(wtpThresholdDistribution  ) # w
        wtpFlexibility = rand(wtpFlexibilityDistribution) # k

        # Create a new customer and add them to the customer list
        customer = Customer(wtpThreshold, wtpFlexibility)
        push!(customersWithoutTicketsByFareClass, customer)

    end

    # Check to see which customers will purchase tickets
    customersWithPurchase = Set{Customer}()

    for customer in deepcopy(customersWithoutTicketsByFareClass)

        # Compute the purchase probability and sample from its Bernoulli distribution
        purchaseProbability = a[f] <= customer.wtpThreshold ? 1 :
                2*ccdf(Logistic(customer.wtpThreshold, customer.wtpFlexibility), a[f]) # ϕ value

        if rand() <= purchaseProbability # the customer wants to buy
            ticketsSold += 1
            push!(customersWithPurchase, customer)
        end

    end

    # Calculate current reward
    revenue = a[f] * ticketsSold

    # Return results
    return ticketsAvailable - ticketsSold, ticketsSold, revenue, newCustomers,
                customersWithoutTicketsByFareClass, customersWithPurchase

end
