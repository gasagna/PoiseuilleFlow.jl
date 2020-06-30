import LinearAlgebra: mul!

export laplacian!, ddx!, ddy!

function laplacian!(∇ψ̂::SpectralField{P, L}, ψ̂::SpectralField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    mul!(∇ψ̂, eq.D², ψ̂)
    for l = 0:L, p = 0:P
        @inbounds  ∇ψ̂.data[p+1, l+1] -= (l*eq.α)^2 * ψ̂.data[p+1, l+1]
    end
    return ∇ψ̂
end

function ddx!(dψ̂dx::SpectralField{P, L}, ψ̂::SpectralField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    for l = 0:L, p = 0:P
        @inbounds dψ̂dx.data[p+1, l+1] = im*l*eq.α*ψ̂[p+1, l+1]
    end
    return dψ̂dx
end

function ddy!(dψ̂dy::SpectralField{P, L}, ψ̂::SpectralField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    mul!(dψ̂dy, eq.D, ψ̂)
    return dψ̂dy
end