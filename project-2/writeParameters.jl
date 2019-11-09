function writeParameters(γ, ϵ, α, λ, f, time, filename)

    open("data/" * filename * ".dat", "w") do io
        @printf(io, "γ    | %f \n", γ)
        @printf(io, "ϵ    | %f \n", ϵ)
        @printf(io, "α    | %f \n", α)
        @printf(io, "λ    | %f \n", λ)
        @printf(io, "f    | %f \n", f)
        @printf(io, "time | %f \n", time)
    end

end
