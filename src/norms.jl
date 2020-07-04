# dot product between two fields in physical space
# we integrate over y using high-order quadrature and in x using a 
# trapezoidal rule, which is exponentially accurate for periodic data
function LinearAlgebra.dot(u::PhysicalField{P, LD}, 
                           v::PhysicalField{P, LD}) where {P, LD}
    w = weights(grid(u))
    out = zero(eltype(u))
    Lx, _ = domain(grid(u))
    @inbounds for l = 1:(2*LD+2)
        @simd for p = 1:P
            out += u[p, l] * v[p, l] * w[p]
        end
    end
    return out*Lx/(2*LD+2)
end

LinearAlgebra.norm(u::PhysicalField) = sqrt(dot(u, u))