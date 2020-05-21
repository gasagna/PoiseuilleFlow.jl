import LinearAlgebra: I, diagm

export chebpoints, chebdiff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
basis_vector(i, P) = (out = zeros(P); out[i] = 1; out)

# ~~~~~ Chebyshev differentiation  ~~~~~~
# P+1 points from 1 to -1
chebpoints(P::Int, ::Type{T}=Float64) where {T} = T.(cos.((0:P)*pi/P))

# chebyshev differentiation matrix
function chebdiff(P::Int, ::Type{T}=Float64) where {T}
    x = chebpoints(P, T)
    c = [2; ones(T, P-1); 2].*(-1).^(0:P)
    X = repeat(x, outer=(1, P+1))
    dX = X - X'
    D = (c*(1 ./ c)')./(dX + I)
    D = D - diagm(0=>vec(sum(D, dims=2)))
    return D
end