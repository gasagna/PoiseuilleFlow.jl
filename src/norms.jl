# dot product between two fields in physical space
function LinearAlgebra.dot(u::PhysicalField{P, LD, Lx, T}, 
                           v::PhysicalField{P, LD, Lx, T}) where {P, LD, Lx, T}
    # we integrate over y using Clenshaw-Curtis quadrature and in x using a 
    # trapezoidal rule, which is exponentially accurate for periodic data
    w = chebweights(Val(P))
    out = zero(T)
    for _l = 1:(2*LD+2)
        for _p = 1:P+1
            out += u[_p, _l] * v[_p, _l] * w[_p]
        end
    end
    return out*Lx/(2*LD+2)
end