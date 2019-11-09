function plotSmall(U, π, πModal, solutionType)

    gr()

    heatmap(reverse(rotl90(reshape(U, (10, 10))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index 1"); ylabel!("Position Index 2")
    savefig("plots/small_U_" * solutionType *".png")

    heatmap(reverse(rotl90(reshape(π, (10, 10))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index 1"); ylabel!("Position Index 2")
    savefig("plots/small_π_" * solutionType *".png")

    heatmap(reverse(rotl90(reshape(πModal, (10, 10))), dims=1),
        c=:viridis, framestyle=:box, dpi=600)
    xlabel!("Position Index 1"); ylabel!("Position Index 2")
    savefig("plots/small_πModal_" * solutionType *".png")

end
