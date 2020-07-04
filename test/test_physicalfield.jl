@testset "physicalfield.jl                       " begin
    # number of points
    P, L, LD = 121, 10, 20

    # grid
    g = Grid(P, L, LD, 1; η=0.5, width=9)

    # size
    u = PhysicalField(g, (x, y)->(x*y)); u .= 1
    @test size(u) == (P, 2LD+2)

    u = PhysicalField(g); u .= 1
    @test size(u) == (P, 2LD+2)

    # similar
    v = similar(u); v .= 2
    w = similar(u); w .= 0

    # broadcasting allocations
    fun(u, v, w) = @allocated w .= 3.0 .* u .+ 2.0 .* v
    @test fun(u, v, w) == 0

    @test all(w .== 7)
end