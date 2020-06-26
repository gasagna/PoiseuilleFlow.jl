@testset "operators                              " begin
    # grid
    P, L = 5, 4
    Lx = 10
    α = 2π / Lx 

    # make grid
    y, x = grid(P, L, Lx)
    
    # construct fields
    ψ      = PhysicalField(P, L, Lx, (x, y) -> cos(α*x) + y^4 + y^3 * sin(2*α*x) )
    dψdy   = PhysicalField(P, L, Lx, (x, y) -> 4*y^3 + 3*y^2 * sin(2*α*x))
    dψdx   = PhysicalField(P, L, Lx, (x, y) -> - α   * sin(α*x) + y^3 * 2*α*cos(2*α*x))
    d²ψdy² = PhysicalField(P, L, Lx, (x, y) -> 12*y^2 + 6*y * sin(2*α*x))
    ∇²ψ    = PhysicalField(P, L, Lx, (x, y) -> 12*y^2 + 6*y * sin(2*α*x)  -α^2 * cos(α*x) - y^3 * (2α)^2 * sin(2*α*x) )
    
    dψ = PhysicalField(P, L, Lx)
    ψ̂  = SpectralField(P, L, Lx)
    dψ̂ = SpectralField(P, L, Lx)
    
    # define transforms
    _fft  = ForwardFFT!(similar(ψ))
    _ifft = InverseFFT!(similar(ψ̂))

    # transform field
    _fft(ψ̂, ψ)
    
    @test all(isapprox.(_ifft(dψ,       ddy!(dψ̂, ψ̂)), dψdy;   atol=1e-12))
    @test all(isapprox.(_ifft(dψ,       ddx!(dψ̂, ψ̂)), dψdx;   atol=1e-12))
    @test all(isapprox.(_ifft(dψ, laplacian!(dψ̂, ψ̂)), ∇²ψ;    atol=1e-12))
    @test all(isapprox.(_ifft(dψ,     d2dy2!(dψ̂, ψ̂)), d²ψdy²; atol=1e-12))
end