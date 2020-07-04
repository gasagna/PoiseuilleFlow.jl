@testset "equations.jl                           " begin

    # number of points
    P, L, LD = 121, 10, 20

    # grid
    g = Grid(P, L, LD, 1; η=0.5, width=7)

    # size
    u = SpectralField(g);
    N = SpectralField(g);

    # should get laminar flow
    Re = 1
    Δt = 1e-2

    # create equations
    eq = StreamFunEquation(g, Re, Δt)

    # create flow operator
    ϕ = flow(eq, eq, CNRK2(u), TimeStepConstant(Δt))

    # @btime $eq(0.0, $û, $N)

    @btime $ϕ($u, (0, 5*$Δt))
    # @time ϕ(û, (0, 1))
    # @time ϕ(û, (0, 1))
    # @time ϕ(û, (0, 1))
 
end
