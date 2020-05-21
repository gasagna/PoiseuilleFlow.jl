using PoiseuilleFlow
using InteractiveUtils
using FFTW
using Test
using Flows

@testset "field.jl                               " begin
    P, L = 10, 20

    # size
    u = Field(P, L); u .= 1
    @test size(u) == (P+1, 2L+1)

    # similar
    v = similar(u); v .= 2
    w = similar(u); w .= 0

    # broadcasting allocations
    fun(u, v, w) = @allocated w .= 3.0 .* u .+ 2.0 .* v
    @test fun(u, v, w) == 0

    @test all(w .== 7)
end


@testset "ftfield.jl                             " begin
    P, L = 10, 20

    # size
    û = FTField(P, L); û .= 1
    @test size(û) == (P+1, L+1)

    # similar
    v̂ = similar(û); v̂ .= 2
    ŵ = similar(û); ŵ .= 0

    # broadcasting allocations
    fun(û, v̂, ŵ) = @allocated ŵ .= 3.0 .* û .+ 2.0 .* v̂
    @test fun(û, v̂, ŵ) == 0

    @test all(ŵ .== 7)

    # indexing
    data = randn(P+1, L+1)
    û .= data
    @test all(parent(û) .== data)
    @test all(û .== data)
    @test û[1, 1] == data[1, 1]
    @test û[P, L] == data[P, L]

    û[:, 2] .= 1
    @test all(parent(û)[:, 2] .== 1)
end


@testset "ffts.jl                                " begin
    P, L, Lx = 5, 5, 1

    # size
    û = FTField(P, L)
    u =   Field(P, L)

    # make transforms
    fft  = ForwardFFT!(u, FFTW.MEASURE)
    ifft = InverseFFT!(û, FFTW.MEASURE)

    # create random data
    data = randn(P+1, 2*L+1)

    u .= data
    fft(û, u)
    ifft(u, û)

    @test maximum(abs, parent(u) .- data) < 1e-14

    # test values
    û .= 0
    û.data[:, 1] .= 2
    ifft(u, û)
    @test all(u.data .== 2)

    y, x = grid(P, L, Lx)
    û .= 0
    û.data[:, 2] .= 0.5
    ifft(u, û)
    @test all(u.data[1, :] .≈  cos.(2π/Lx * x))

    û .= 0
    û.data[:, 3] .= -0.5*im
    ifft(u, û)
    @test all(u.data[1, :] .≈ sin.(2*2π/Lx * x))

end

@testset "operators.jl                           " begin
    P, L = 3, 3
    Lx = 10
    Re = 1

    # create fields
    û  = FTField(P, L)
    dû = FTField(P, L)

    # create equations
    eq = StreamFunEquation(P, L, Lx, Re, 1e-2)

    # set data for x direction
    û .= 0
    û[:, 2+1] .= 1
    ddx!(dû, û, eq)
    @test all(dû[:, 2+1] .== im*2*2π/Lx)

    # set data for y direction
    y = chebpoints(P)
    û .= 0
    û[:, 0+1] .= (1 .- y.^2)
    ddy!(dû, û, eq)
    @test maximum(abs, dû[:, 0+1] .+ 2 .* y) < 1e-14

    # set data for laplacian
    y = chebpoints(P)
    û .= 0
    û[:, 3+1] .= (1 .- y.^2)
    laplacian!(dû, û, eq)
    @test maximum(abs, dû[:, 3+1] .- (.- 2 .- (3*2π/Lx)^2.0*(1 .- y.^2))) < 1e-14
end

@testset "equations.jl                           " begin

    P, L = 20, 20
    Lx = 10
    Re = 1
    Δt = 1e-2

    # create fields
    û  = FTField(P, L)
    dû = FTField(P, L)

    # create equations
    eq = StreamFunEquation(P, L, Lx, Re, Δt)

    # create flow operator
    ϕ = flow(eq, eq, CNRK2(û), TimeStepConstant(Δt))

    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
    @time ϕ(û, (0, 1))
    # println(@code_warntype ddy!(dû, û, eq))
    # println(methods(Flows.ImcA_mul!))
end