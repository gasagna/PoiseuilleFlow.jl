export PhysicalField, grid

struct PhysicalField{P, LD, T, G} <: AbstractMatrix{T}
    data::Matrix{T} # actual data
    grid::G         # grid object for dot products, norms
    function PhysicalField(grid::Grid{P, L, LD}, ::Type{T}=Float64) where {T<:Real, P, L, LD}
        data = zeros(T, P, 2*LD+2)
        return new{P, LD, T, typeof(grid)}(data, grid)
    end

    # construct from a function accepting x and y
    function PhysicalField(grid::Grid{P, L, LD}, fun::F, ::Type{T}=Float64) where {T<:Real, P, L, LD, F}
        x, y = points(grid)
        data = fun.(x', y)
        return new{P, LD, T, typeof(grid)}(data, grid)
    end
end

# get the grid
grid(f::PhysicalField) = f.grid

Base.parent(f::PhysicalField) = f.data
Base.size(::PhysicalField{P, LD}) where {P, LD} = (P, 2LD+2)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()
Base.similar(u::PhysicalField) = PhysicalField(grid(u), eltype(u))

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

save(filename::String, ψ::PhysicalField) =
    (writedlm(filename, parent(ψ)); nothing)

function load(filename::String, g::Grid)
    data = readdlm(filename)
    P, LL = size(data)
    LD = div(LL, 2) - 1
    ψ = PhysicalField(grid, eltype(data))
    parent(ψ) .= data
    return ψ
end