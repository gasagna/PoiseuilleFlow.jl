@testset "operators                              " begin

    P, L = 3, 3
    Lx = 10
    Re = 1

    # create fields
    û   = FTField(P, L)
    dû  = FTField(P, L)
    d2û = FTField(P, L)

    # # set data for x direction
    # û .= 0
    # û[:, 2+1] .= 1
    # ddx!(dû, û, eq)
    # @test all(dû[:, 2+1] .== im*2*2π/Lx)

    # set data for y direction
    û .= 0

    # set the polinomial x
    for l = 0:L
        û[1, l] = 1
    end

    ddy!(dû, û)
    d2dy2!(d2û, û)

    # the derivative is one
    for l = 0:L
        @test dû[ 0, l] == 1
        @test d2û[0, l] == 0
    end
    
    # FIXME: add when we have transforms available

    # û[:, 0+1] .= (1 .- y.^2)
    # ddy!(dû, û, eq)
    # @test maximum(abs, dû[:, 0+1] .+ 2 .* y) < 1e-14

    # # set data for laplacian
    # y = chebpoints(P)
    # û .= 0
    # û[:, 3+1] .= (1 .- y.^2)
    # laplacian!(dû, û, eq)
    # @test maximum(abs, dû[:, 3+1] .- (.- 2 .- (3*2π/Lx)^2.0*(1 .- y.^2))) < 1e-14

end