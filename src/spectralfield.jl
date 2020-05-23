using DelimitedFiles

export SpectralField

struct SpectralField{P, L, Lx, T} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}}
    function SpectralField(P::Int, L::Int, Lx::Real, ::Type{T}=Float64) where {T<:Real}
        data = zeros(Complex{T}, P+1, L+1)
        return new{P, L, Lx, T}(data)
    end
end

Base.parent(f::SpectralField) = f.data
Base.size(::SpectralField{P, L}) where {P, L} = (P+1, L+1)
# FIXME:
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.similar(u::SpectralField{P, L, Lx, T}) where {P, L, Lx, T} = SpectralField(P, L, Lx, T)

@inline reindex(p, l) = (p+1, l+1)
@inline reindex(i) = i

# linear indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::SpectralField{P, L}, i...) where {P, L}
    # FIXME:
    # @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds ret = parent(ψ̂)[reindex(i...)...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::SpectralField{P, L}, v, i...) where {P, L}
    # FIXME:
    # @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds parent(ψ̂)[reindex(i...)...] = v
    return v
end

# save(filename::String, u::SpectralField) =
#     (writedlm(filename, vcat(real.(parent(u)), imag.(parent(u))); nothing)

# # function load(filename::String)
# #     data = readdlm(filename)
# #     PP, LL = size(data)
# #     P = div(PP, 2)-1
# #     L = LL - 1
# #     u = SpectralField(P, L)
# #     parent(u) .= 
# # end