@testset "ftfield.jl                             " begin
    P, L, Lx = 11, 20, 1

    # P must be odd
    @test_throws ArgumentError SpectralField(P+1, L, Lx)

    # size
    û = SpectralField(P, L, Lx); û .= 1
    @test size(û) == (P+1, L+1)
    @test all(û.data .== 1)

    # similar
    v̂ = similar(û); v̂ .= 2
    ŵ = similar(û); ŵ .= 0

    # broadcasting allocations
    fun(û, v̂, ŵ) = @allocated ŵ .= 3.0 .* û .+ 2.0 .* v̂
    @test fun(û, v̂, ŵ) == 0

    # broadcasting value
    @test all(ŵ .== 7)

    # indexing
    data = randn(P+1, L+1)
    û .= data
    @test all(parent(û) .== data)
    @test all(û .== data)
    @test û[0, 0] == data[1, 1]
    @test û[P, L] == data[P+1, L+1]

    û[:, 2] .= 1
    @test all(parent(û)[:, 2] .== 1)
end