using DelimitedFiles

export SpectralField

# e.g. LD = 3 -> 2*3 + 2 = 8 grid points
# LD is the highest wavenumber, excluding the nyquist frequency
# the size of the transform given LD is LD + 2, so this is the size of the array
# L is the highest resolved frequency
struct SpectralField{P, L, LD, T} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}}
    function SpectralField(P::Int, L::Int, LD::Int, ::Type{T}=Float64) where {T<:Real}
        data = zeros(Complex{T}, P+1, LD+2)
        return new{P, L, LD, T}(data)
    end
end

Base.parent(f::SpectralField) = f.data
Base.size(::SpectralField{P, L, LD}) where {P, L, LD} = (P+1, L+1)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.similar(u::SpectralField{P, L, LD, T}) where {P, L, LD, T} = SpectralField(P, L, LD, T)

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

# save(filename::String, u::SpectralField) =
#     (writedlm(filename, vcat(real.(parent(u)), imag.(parent(u))); nothing)

# function load(filename::String)
#     data = readdlm(filename)
#     PP, LL = size(data)
#     P = div(PP, 2)-1
#     L = LL - 1
#     u = SpectralField(P, L)
#     parent(u) .= 
# end