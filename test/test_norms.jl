@testset "norms.jl                               " begin
    # grid
    P, L, LD, Lx = 201, 10, 20, 4

    # grid
    g = Grid(P, L, LD, Lx; η=0.5, width=11)

    # fundamental wavenumber
    α  = 2π/Lx

    # dot product
    for (fun_u, fun_v, res) in (
                                ((x, y)->(1+cos(π*y))*cos(α*x), (x, y)->(1+cos(π*y))*cos(α*x),  6), 
                               )
        u = PhysicalField(g, fun_u)
        v = PhysicalField(g, fun_v)
        @test dot(u, v) ≈ res
    end

    # norm
    u = PhysicalField(g, (x, y)->(1+cos(π*y))*cos(α*x))
    @test norm(u) ≈ sqrt(6)
end