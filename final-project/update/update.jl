using Distributions
using Random
using StatsPlots
using Plots
using Colors


function plotCustomerPriceSchedule()

    customerPriceThreshold = Normal(400, 25)
    customerPriceStrength  = Uniform(1, 10^1.2)

    cpt = rand(customerPriceThreshold, 50)
    cps = rand(customerPriceStrength,  50)

    customerPriceSchedule = Logistic.(cpt, cps)

    plot()

    for i in 1:size(customerPriceSchedule)[1]
        plot!(200:0.5:600, ccdf.(customerPriceSchedule[i], 200:0.5:600),
        legend=:none, color=RGB(69/255, 165/255, 245/255), alpha=0.4, box=:on,
        widen=:false, xlabel="Price (\$)", ylabel="Purchase Likelihood", dpi=600)
    end
    plot!([200, 600], [0.5, 0.5], linestyle=:dash, color=:gray)
    plot!(200:600, 10*pdf.(customerPriceThreshold, 200:600).+0.5, color=:gray, alpha=0.6)
    scatter!(cpt, 0.5*ones(50), color=:gray, alpha=0.2)
    plot!(dpi=600)

end

plotCustomerPriceSchedule()
savefig("customer_price_schedule.png")

function plotCustomerArrival()

    H = 20
    ca = zeros(50, H)

    plot()
    for i in 1:50
        for t in 1:H
            customerArrival = Poisson(3*(H-t)+10)
            ca[i,t] = Float64(rand(customerArrival, 1)...)
        end

        plot!(1:H, ca[i,:], legend=:none, color=RGB(69/255, 165/255, 245/255),
            alpha=0.4, box=:on, widen=:false, xlabel="Days", ylabel="New Customers Arriving")
        scatter!(1:H, ca[i,:], legend=:none, color=:gray, alpha=0.2, box=:on, widen=:false)

    end
    plot!()

end

plotCustomerArrival()
savefig("customer_arrival.png")
