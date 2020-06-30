@testset "field.jl                               " begin
    P  = 31
    LD = 80
    Lx = 4
    α  = 2π/Lx

    for (fun_u, fun_v, res) in (
                                ((x, y)->(1+cos(π*y))*cos(α*x), (x, y)->(1+cos(π*y))*cos(α*x),  6), 
                               )
        u = PhysicalField(P, LD, Lx, fun_u)
        v = PhysicalField(P, LD, Lx, fun_v)
        @test dot(u, v) ≈ res
    end
end