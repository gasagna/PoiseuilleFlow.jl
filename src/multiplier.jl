struct Multiplier{T, F, M}
    tmp::F
    mat::M
    function Multiplier(mat::AbstractMatrix{T}) where {T}
        tmp = ntuple(i->zeros(T, size(mat, 1)), 4)
        return new{T, typeof(tmp), typeof(mat)}(tmp, mat)
    end
end

function LinearAlgebra.mul!(y::SpectralField{P, L}, 
                            D::Multiplier,
                            x::SpectralField{P, L}) where {P, L}
    tmp1_re, tmp2_re, tmp1_im, tmp2_im = D.tmp
    for l = 0:L
        _copycol!(tmp1_re, tmp1_im, x.data, l+1)
        mul!(tmp2_re, D.mat, tmp1_re)
        mul!(tmp2_im, D.mat, tmp1_im)
        _copycol!(y.data, tmp2_re, tmp2_im, l+1)
    end
    return y
end


# two column version
function _copycol!(v_re::AbstractVector, v_im::AbstractVector, M::AbstractMatrix, c::Int)
    length(v_re) == length(v_im) == size(M, 1) || throw(ArgumentError("invalid size"))
    @inbounds @simd for i = 1:length(v_re)
        _re, _im = reim(M[i, c])
        v_re[i] = _re
        v_im[i] = _im
    end
    return nothing
end

function _copycol!(M::AbstractMatrix, v_re::AbstractVector, v_im::AbstractVector, c::Int)
    length(v_re) == length(v_im) == size(M, 1) || throw(ArgumentError("invalid size"))
    @inbounds @simd for i = 1:length(v_re)
        M[i, c] = v_re[i] + im*v_im[i]
    end
    return nothing
end

# single column version
function _copycol!(v::AbstractVector, M::AbstractMatrix, c::Int)
    length(v) == size(M, 1) || throw(ArgumentError("invalid size"))
    for i = 1:length(v)
        @inbounds v[i] = M[i, c]
    end
    return nothing
end

function _copycol!(M::AbstractMatrix, v::AbstractVector, c::Int)
    length(v) == size(M, 1) || throw(ArgumentError("invalid size"))
    for i = 1:length(v)
        @inbounds M[i, c] = v[i]
    end
    return nothing
end