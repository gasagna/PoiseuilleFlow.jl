import LinearAlgebra: mul!

export laplacian!, ddx!, ddy!

function laplacian!(∇ψ̂::SpectralField{P, L}, ψ̂::SpectralField{P, L}) where {P, L}
    mul!(∇ψ̂, grid(ψ̂).Ds[2], ψ̂)
    α = 2π/domain(grid(ψ̂))[1]
    for l = 0:L, p = 1:P
        @inbounds  ∇ψ̂.data[p, l+1] -= (l*α)^2 * ψ̂.data[p, l+1]
    end
    return ∇ψ̂
end

function ddx!(dψ̂dx::SpectralField{P, L}, ψ̂::SpectralField{P, L}) where {P, L}
    α = 2π/domain(grid(ψ̂))[1]
    for l = 0:L, p = 1:P
        @inbounds dψ̂dx.data[p, l+1] = im*l*α*ψ̂[p, l+1]
    end
    return dψ̂dx
end

ddy!(dψ̂dy::SpectralField, ψ̂::SpectralField) = 
    mul!(dψ̂dy, grid(ψ̂).Ds[1], ψ̂)