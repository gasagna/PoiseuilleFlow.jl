@testset "ffts.jl                                " begin
    # number of points
    P, L, LD = 121, 20, 20

    # grid
    g = Grid(P, L, LD, 1; η=0.5, width=9)

    # size
    u = PhysicalField(g)
    U = SpectralField(g)

     _fft = ForwardFFT!(u)
    _ifft = InverseFFT!(U)

    # set data
    U.data[:, 1:10] .= rand(P, 10)

    # backward
    _ifft(u, U)

    # forward
    V = _fft(copy(U), u)

    @test maximum(abs, U.data - V.data) < 1e-15
end