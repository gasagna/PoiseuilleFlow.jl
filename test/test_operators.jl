# @testset "operators.jl                           " begin
#     P, L = 3, 3
#     Lx = 10
#     Re = 1

#     # create fields
#     û  = FTField(P, L)
#     dû = FTField(P, L)

#     # create equations
#     eq = StreamFunEquation(P, L, Lx, Re, 1e-2)

#     # set data for x direction
#     û .= 0
#     û[:, 2+1] .= 1
#     ddx!(dû, û, eq)
#     @test all(dû[:, 2+1] .== im*2*2π/Lx)

#     # set data for y direction
#     y = chebpoints(P)
#     û .= 0
#     û[:, 0+1] .= (1 .- y.^2)
#     ddy!(dû, û, eq)
#     @test maximum(abs, dû[:, 0+1] .+ 2 .* y) < 1e-14

#     # set data for laplacian
#     y = chebpoints(P)
#     û .= 0
#     û[:, 3+1] .= (1 .- y.^2)
#     laplacian!(dû, û, eq)
#     @test maximum(abs, dû[:, 3+1] .- (.- 2 .- (3*2π/Lx)^2.0*(1 .- y.^2))) < 1e-14
# end