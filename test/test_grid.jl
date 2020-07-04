@testset "grid.jl                                " begin
    # nuber of points
    P, L, LD = 121, 5, 10
    
    # grid
    g = Grid(P, L, LD, 1; η=0.5, width=9)

    # points
    x, y = points(g)

    # domain size
    @test domain(g) == (1, 2)

    # grid size
    @test size(g, :spectral) == (P, L)

    # test integration
    @test abs(sum(ones(P) .* weights(g)) - 2) < 1e-15
    @test abs(sum(      y .* weights(g)) - 0) < 1e-15
end