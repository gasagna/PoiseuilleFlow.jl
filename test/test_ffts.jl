@testset "ffts.jl                                " begin
    P, L, Lx = 5, 5, 1

    # size
    û = SpectralField(P, L, Lx)
    u = PhysicalField(P, L, Lx)

    # make transforms
    fft  = ForwardFFT!(u, FFTW.ESTIMATE)
    ifft = InverseFFT!(û, FFTW.ESTIMATE)

    # create random data
    data = randn(P+1, 2*L+1)

    u .= data
    fft(û, u)
    ifft(u, û)

    @test maximum(abs, parent(u) .- data) < 1e-14

    # this represents the field cos(2π/Lx * x) * T_4(y)
    û .*= 0
    û[4, 1] = 0.25 # note it's one fourth
    ifft(u, û)

    # obtain grid
    y, x = grid(P, L, Lx)

    # check values
    for (i, xi) in enumerate(x)
        @test maximum(abs, u[:, i] .- cos(2π/Lx*xi)*PoiseuilleFlow.T4.(y)) < 1e-15
    end

    for (i, xi) in enumerate(x)
        u[:, i] .= (0.4 .* PoiseuilleFlow.T1.(y)*cos(2*2π/Lx*xi) .+ 
                    0.2 .* PoiseuilleFlow.T2.(y)*sin(1*2π/Lx*xi))
    end
    fft(û, u)
    @test û[1, 2] ≈  0.10
    @test û[2, 1] ≈ -0.05*im
end