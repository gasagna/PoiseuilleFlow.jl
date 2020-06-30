export PhysicalField, grid

struct PhysicalField{P, LD, Lx, T} <: AbstractMatrix{T}
    data::Matrix{T}
    function PhysicalField(P::Int, LD::Int, Lx::Real, ::Type{T}=Float64) where {T<:Real}
        data = zeros(T, P+1, 2*LD+2)
        return new{P, LD, Lx, T}(data)
    end

    # construct from a function accepting x and y
    function PhysicalField(P::Int, LD::Int, Lx::Real, fun::F, ::Type{T}=Float64) where {F, T<:Real}
        y, x = grid(P, LD, Lx)
        data = fun.(x', y)
        return new{P, LD, Lx, T}(data)
    end
end

Base.parent(f::PhysicalField) = f.data
Base.size(::PhysicalField{P, LD}) where {P, LD} = (P+1, 2LD+2)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()
Base.similar(u::PhysicalField{P, LD, Lx, T}) where {P, LD, Lx, T} = PhysicalField(P, LD, Lx, T)

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

grid(P::Int, LD::Int, Lx::Real) = chebpoints(P), (0:(2LD+1))/(2LD+2)*Lx

save(filename::String, ψ::PhysicalField) =
    (writedlm(filename, parent(ψ)); nothing)

function load(filename::String, Lx::Real)
    data = readdlm(filename)
    PP, LL = size(data)
    P = PP - 1
    LD = div(LL, 2) - 1
    ψ = PhysicalField(P, LD, Lx, eltype(data))
    parent(ψ) .= data
    return ψ
end