function plotLarge(U, π, solutionType, uniqueStates)

    gr()

    U = rotl90(reshape(U, (10000, 32)))
    heatmap(reverse(U[:, sort(unique(mod.(uniqueStates, 10000)))], dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Inferred Index 1"); ylabel!("Inferred Index 2")
    savefig("plots/large_U_" * solutionType *".png")

    π = rotl90(reshape(π, (10000, 32)))
    heatmap(reverse(π[:, sort(unique(mod.(uniqueStates, 10000)))], dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Inferred Index 1"); ylabel!("Inferred Index 2")
    savefig("plots/large_π_" * solutionType *".png")

end
