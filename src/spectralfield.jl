using DelimitedFiles

export SpectralField

struct SpectralField{P, L, T} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}}
    function SpectralField(P::Int, L::Int, ::Type{T}=Float64) where {T<:Real}
        data = zeros(Complex{T}, P+1, L+1)
        return new{P, L, T}(data)
    end
end

Base.parent(f::SpectralField) = f.data
Base.size(::SpectralField{P, L}) where {P, L} = (P+1, L+1)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.similar(u::SpectralField{P, L, T}) where {P, L, T} = SpectralField(P, L, T)

# linear indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::SpectralField{P, L}, i...) where {P, L}
    @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds ret = parent(ψ̂)[i...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::SpectralField{P, L}, v, i...) where {P, L}
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