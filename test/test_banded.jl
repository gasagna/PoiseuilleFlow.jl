@testset "QuasiTridiagonal                       " begin

    # full matrix
    B = Float64[4 4 4 4 4;
                3 5 1 0 0;
                0 3 6 2 0;
                0 0 3 7 3;
                0 0 0 3 8;]

    l = Float64[3, 3, 3, 3]
    d = Float64[5, 6, 7, 8]
    u = Float64[1, 2, 3]
    b = Float64[4, 4, 4, 4, 4]

    # define object
    A = QuasiTridiagonal(b, l, d, u)

    # test elements
    @test A == B

    # right hand size
    c = Float64[1, 2, 3, 4, 5]
    
    # factorise and solve
    ul!(A)

    # extract factors
    U = triu(A)
    L = tril(A); for i = 1:size(L, 1); L[i, i] = 1; end

    # test decomposition
    @test norm(U*L - B) < 1e-15

    # solve
    x = ldiv!(A, copy(c))

    # exact solution
    x_ex = B\c
    
    @test norm(x_ex - x) < 1e-15
end