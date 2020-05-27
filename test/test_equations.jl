@testset "compatibility with Flow.jl             " begin

    P, L, Lx = 21, 20, 10
    Re = 1
    Δt = 1e-2

    # create fields
    û  = SpectralField(P, L, Lx)

    # create equations
    eq = StreamFunEquation(P, L, Lx, Re)

    # create flow operator
    ϕ = flow(eq, eq, CNRK2(û), TimeStepConstant(Δt))

    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
end

