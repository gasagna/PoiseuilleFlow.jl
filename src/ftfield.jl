using DelimitedFiles

export FTField

struct FTField{P, L, T} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}}
    function FTField(P::Int, L::Int, ::Type{T}=Float64) where {T<:Real}
        data = zeros(Complex{T}, P+1, L+1)
        return new{P, L, T}(data)
    end
end

Base.parent(f::FTField) = f.data
Base.size(::FTField{P, L}) where {P, L} = (P+1, L+1)
Base.IndexStyle(::Type{<:FTField}) = Base.IndexLinear()
Base.similar(u::FTField{P, L, T}) where {P, L, T} = FTField(P, L, T)

# linear indexing
Base.@propagate_inbounds function Base.getindex(ψ̂::FTField{P, L}, i...) where {P, L}
    @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds ret = parent(ψ̂)[i...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(ψ̂::FTField{P, L}, v, i...) where {P, L}
    @boundscheck checkbounds(parent(ψ̂), i...)
    @inbounds parent(ψ̂)[i...] = v
    return v
end

save(filename::String, u::FTField) =
    (writedlm(filename, vcat(real.(parent(u)), imag.(parent(u))); nothing)

function load(filename::String)
    data = readdlm(filename)
    PP, LL = size(data)
    P = div(PP, 2)-1
    L = LL - 1
    u = FTField(P, L)
    parent(u) .= 
end