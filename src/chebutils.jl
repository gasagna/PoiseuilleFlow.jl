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

# Compute Clenshaw-Curtis quadrature weights 
# see https://people.math.ethz.ch/~joergw/Papers/fejer.pdf
# so that sum(w*f) is the integral of f sampled at the
# extremes of chebyshev polynomials
function chebweights(P::Int)
    w = zeros(P+1)
    for k = 0:P
        v = 0.0
        θₖ = k*π/P
        for j = 1:div(P, 2)
            bⱼ = j == div(P, 2) ? 1 : 2
            v += bⱼ / (4j^2 - 1) * cos(2*j*θₖ)
        end
        cₖ = k ∈ (0, P) ? 1 : 2
        w[k+1] = cₖ/P *(1 - v)
    end
    return w
end 

@generated function chebweights(::Val{P}) where {P}
    return quote
        return $(tuple(chebweights(P)...))
    end
end 