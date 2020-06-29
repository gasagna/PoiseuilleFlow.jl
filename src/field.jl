export Field, grid

struct Field{P, L, T} <: AbstractMatrix{T}
    data::Matrix{T}
    function Field(P::Int, L::Int, ::Type{T}=Float64) where {T<:Real}
        data = zeros(T, P+1, 2*L+1)
        return new{P, L, T}(data)
    end

    # construct from a function accepting x and y
    function Field(P::Int, L::Int, Lx::Real, fun::F, ::Type{T}=Float64) where {F, T<:Real}
        y, x = grid(P, L, Lx)
        data = fun.(x', y)
        return new{P, L, T}(data)
    end
end

Base.parent(f::Field) = f.data
Base.size(::Field{P, L}) where {P, L} = (P+1, 2L+1)
Base.IndexStyle(::Type{<:Field}) = Base.IndexLinear()
Base.similar(u::Field{P, L, T}) where {P, L, T} = Field(P, L, T)

Base.@propagate_inbounds function Base.getindex(ψ::Field, I...)
    @boundscheck checkbounds(parent(ψ), I...)
    @inbounds ret = parent(ψ)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ::Field, v, I...)
    @boundscheck checkbounds(parent(ψ), I...)
    @inbounds parent(ψ)[I...] = v
    return v
end

grid(P::Int, L::Int, Lx::Real) = chebpoints(P), (0:(2L))/(2L+1)*Lx