export PhysicalField, grid

struct PhysicalField{P, L, Lx, T} <: AbstractMatrix{T}
    data::Matrix{T}
    function PhysicalField(P::Int, L::Int, Lx::Real, ::Type{T}=Float64) where {T<:Real}
        data = zeros(T, P+1, 2*L+1)
        return new{P, L, Lx, T}(data)
    end
end

Base.parent(f::PhysicalField) = f.data
Base.size(::PhysicalField{P, L}) where {P, L} = (P+1, 2L+1)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()
Base.similar(u::PhysicalField{P, L, Lx, T}) where {P, L, Lx, T} = PhysicalField(P, L, Lx, T)

Base.@propagate_inbounds function Base.getindex(ψ::PhysicalField, I...)
    @boundscheck checkbounds(parent(ψ), I...)
    @inbounds ret = parent(ψ)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ::PhysicalField, v, I...)
    @boundscheck checkbounds(parent(ψ), I...)
    @inbounds parent(ψ)[I...] = v
    return v
end

grid(P::Int, L::Int, Lx::Real) = chebpoints(P), (0:(2L))/(2L+1)*Lx