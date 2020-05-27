@testset "physicalfield                          " begin
    P, L, Lx = 11, 20, 1

    # P must be odd
    @test_throws ArgumentError PhysicalField(P+1, L, Lx)
    
    # size
    u = PhysicalField(P, L, Lx); u .= 1
    @test size(u) == (P+1, 2L+1)

    # similar
    v = similar(u); v .= 2
    w = similar(u); w .= 0

    # broadcasting allocations
    fun(u, v, w) = @allocated w .= 3.0 .* u .+ 2.0 .* v
    @test fun(u, v, w) == 0

    @test all(w .== 7)
    @test all(parent(w) .== 7)
end