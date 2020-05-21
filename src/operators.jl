import LinearAlgebra: mul!

export laplacian!, ddx!, ddy!

function laplacian!(∇ψ̂::FTField{P, L}, ψ̂::FTField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    mul!(∇ψ̂.data, eq.D², ψ̂.data)
    for l = 0:L, p = 0:P
        @inbounds  ∇ψ̂.data[p+1, l+1] -= (l*eq.α)^2 * ψ̂.data[p+1, l+1]
    end
    return ∇ψ̂
end

function ddx!(dψ̂dx::FTField{P, L}, ψ̂::FTField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    for l = 0:L, p = 0:P
        @inbounds dψ̂dx.data[p+1, l+1] = im*l*eq.α*ψ̂[p+1, l+1]
    end
    return dψ̂dx
end

function ddy!(dψ̂dy::FTField{P, L}, ψ̂::FTField{P, L}, eq::StreamFunEquation{P, L}) where {P, L}
    mul!(dψ̂dy.data, eq.D, ψ̂.data)
    return dψ̂dy
end