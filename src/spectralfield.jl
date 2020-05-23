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
Base.size(::SpectralField{P, L}) where {P, L} = (P+1, L+1) #(0:(P+1), 0:(L+1))
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.similar(u::SpectralField{P, L, Lx, T}) where {P, L, Lx, T} = SpectralField(P, L, Lx, T)

@inline reindex(p, l) = (p+1, l+1)
@inline reindex(i) = i

# linear indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::SpectralField, i::Int)
    @boundscheck checkbounds(parent(ψ̂), i)
    @inbounds ret = parent(ψ̂)[i]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::SpectralField, v, i::Int)
    @boundscheck checkbounds(parent(ψ̂), i)
    @inbounds parent(ψ̂)[i] = v
    return v
end

# cartesian indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::SpectralField, p::Int, l::Int)
    @boundscheck checkbounds(parent(ψ̂), p+1, l+1)
    @inbounds ret = parent(ψ̂)[p+1, l+1]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::SpectralField, v, p::Int, l::Int)
    @boundscheck checkbounds(parent(ψ̂), p+1, l+1)
    @inbounds parent(ψ̂)[p+1, l+1] = v
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