import LinearAlgebra: mul!

export laplacian!, ddx!, ddy!, d2dy2!

function laplacian!(∇ψ̂::FTField{P, L, Lx}, ψ̂::FTField{P, L, Lx}) where {P, L, Lx}
    d2dy2!(∇ψ̂, ψ̂)
    @inbounds @avx for l = 0:L, p = 0:P
        ∇ψ̂[p, l] -= (l*2π/Lx)^2 * ψ̂[p, l]
    end
    return ∇ψ̂
end

function ddx!(dψ̂dx::FTField{P, L, Lx}, ψ̂::FTField{P, L, Lx}) where {P, L, Lx}
    @inbounds @avx for l = 0:L, p = 0:P
        dψ̂dx[p, l] = im*l*2π/Lx*ψ̂[p, l]
    end
    return dψ̂dx
end

# Compute first wall-normal derivative of spectral field `û` and store it in `dûdy`.
function ddy!(dûdy::FTField{P, L}, û::FTField{P, L}) where {P, L}
    # Equation 2.4.25 of CHQZ
    @inbounds begin
        for l = 0:L
            dûdy[P, l]   = 0
            dûdy[P-1, l] = 2*P*û[P, l]
            for k = reverse(0:P-2)
                dûdy[k, l] = (û[k+2, l] + 2*(k+1)*û[k+1, l])
            end
            dûdy[0, l] *= 0.5
        end
    end
    return dûdy
end

# Compute second wall-normal derivative of spectral field `û` and store it in `dûdy`.
function d2dy2!(d2ûdy2::FTField{P, L}, û::FTField{P, L}) where {P, L}
    @inbounds begin
        for l = 0:L
            d2ûdy2[P, l]   = 0
            d2ûdy2[P-1, l] = 0
            # compute the first derivative, but only store what's required to compute
            # the second derivative. This is faster than using ddy! twice.
            dûdy_p0, dûdy_p1, dûdy_p2 = 2*P*û[P, l], zero(eltype(û)), zero(eltype(û))
            for k = reverse(0:P-2)
                # second derivative
                d2ûdy2[k, l] = d2ûdy2[k+2, l] + 2*(k+1)*dûdy_p1

                # update first derivative
                dûdy_p0, dûdy_p1, dûdy_p2 = (dûdy_p2 + 2*(k+1)*û[k+1, l]), dûdy_p0, dûdy_p2
            end
            d2ûdy2[0, l] *= 0.5
        end
    end
    return d2ûdy2
end