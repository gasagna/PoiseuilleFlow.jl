import LinearAlgebra

export QuasiTridiagonal, ul!

# Type to store a quasi-tridiagonal matrix structured as follows
#  b | b b b b 
#  - + - - - - 
#  l | d u     
#    | l d u   
#    |   l d u 
#    |     l d 
# of size (M, M) with element type T. The type has a fast O(M) UL factorisation 
# without pivoting that preserves the structure (the LU decomposition would not
# do so) and a fast solver based on backward/forward substitution scaling as O(M)
struct QuasiTridiagonal{T, M} <: AbstractMatrix{T}
    b::Vector{T} # length M   # top row
    l::Vector{T} # length M-1 # lower diagonal
    d::Vector{T} # length M-1 # diagonal
    u::Vector{T} # length M-2 # upper diagonal
    function QuasiTridiagonal(b::Vector{T}, l::Vector{T}, d::Vector{T}, u::Vector{T}) where {T}
        M = length(b)
        (length(l) == M-1) && (length(d) == M-1) && (length(u) == M-2) ||
            throw(ArgumentError("incompatible lengths")) 
        return new{T, M}(b, l, d, u)
    end
end

Base.size(Q::QuasiTridiagonal{T, M}) where {T, M} = (M, M)

Base.@propagate_inbounds function Base.getindex(Q::QuasiTridiagonal{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(Q, i, j)
    i == 1   && return Q.b[j]
    i == j   && return Q.d[i-1]
    i == j-1 && return Q.u[i-1]
    i == j+1 && return Q.l[i-1]
    return zero(T)
end

# Compute in place UL factorisation. It is the user responsibility to 
# make sure the matix is factorised before being used for solving a linear 
# sytem.
# 
# I have tried understanding Gibson's channelflow code and digging around 
# for a reference that shows this algorithm. Canuto and other references 
# mention how it's done, but I had to derive the UL factorisation algorithm
# myself.
function ul!(Q::QuasiTridiagonal{T, M}) where {T, M}
    l, d, u, b = Q.l, Q.d, Q.u, Q.b
    @inbounds begin
        l[M-1] = l[M-1]/d[M-1]
        b[M-1] = b[M-1] - b[M]*l[M-1]
        @simd for i = reverse(1:M-2)
            d[i] = d[i] - u[i]*l[i+1] 
            l[i] = l[i]/d[i]
            b[i] = b[i] - b[i+1]*l[i]
        end
    end
    return Q
end

# Solve a linear system Q*x = c in place, overwriting c with solution x.
# Assumes A has been already factorised. 
function LinearAlgebra.ldiv!(Q::QuasiTridiagonal{T, M}, c::Vector{T}) where {T, M}
    # aliases
    l, u, d, b = Q.l, Q.u, Q.d, Q.b
    @inbounds begin
        # backward substiturion
        c[M] = c[M]/d[M-1]
        Σ = c[M]*b[M]
        @simd for k = reverse(2:M-1)
            c[k] = (c[k] - u[k-1]*c[k+1])/d[k-1]
            Σ += c[k]*b[k]
        end
        c[1] = (c[1] - Σ)/b[1]

        # forward substitution
        @simd for k = 2:M
            c[k] = c[k] - c[k-1]*l[k-1]
        end
    end
    return c
end
