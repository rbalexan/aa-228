function plotMedium(U, π, πModal, solutionType)

    gr()

    heatmap(reverse(rotl90(reshape(U, (500, 100))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index"); ylabel!("Velocity Index")
    savefig("plots/medium_U_" * solutionType *".png")

    heatmap(reverse(rotl90(reshape(π, (500, 100))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index"); ylabel!("Velocity Index")
    savefig("plots/medium_π_" * solutionType *".png")

    heatmap(reverse(rotl90(reshape(πModal, (500, 100))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index"); ylabel!("Velocity Index")
    savefig("plots/medium_πModal_" * solutionType *".png")

end
