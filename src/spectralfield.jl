using DelimitedFiles

export SpectralField, grid

# e.g. LD = 3 -> 2*3 + 2 = 8 grid points
# LD is the highest wavenumber, excluding the nyquist frequency
# the size of the transform given LD is LD + 2, so this is the size of the array
# L is the highest resolved frequency
struct SpectralField{P, L, LD, T, G} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}} # actual data
    grid::G                  # grid object for dot products, norms
    function SpectralField(grid::Grid{P, L, LD}, ::Type{T}=Float64) where {T<:Real, P, L, LD}
        data = zeros(Complex{T}, P, LD+2)
        return new{P, L, LD, T, typeof(grid)}(data, grid)
    end
end

# get the grid
grid(f::SpectralField) = f.grid

Base.parent(f::SpectralField) = f.data
Base.size(::SpectralField{P, L}) where {P, L} = (P, L+1)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.similar(u::SpectralField{P, L, LD, T}) where {P, L, LD, T} = SpectralField(grid(u), T)
Base.copy(u::SpectralField{P, L, LD, T}) where {P, L, LD, T} = 
    (v = SpectralField(grid(u), T); v .= u; v)

# linear indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::SpectralField, i...)
    @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds ret = parent(ψ̂)[i...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::SpectralField, v, i...)
    @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds parent(ψ̂)[i...] = v
    return v
end