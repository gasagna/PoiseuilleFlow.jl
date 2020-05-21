import LinearAlgebra: I, lu!, diagm, mul!, ldiv!
import Flows

export StreamFunEquation

struct StreamFunEquation{P, L, V1, V2, NT, FT, IFT, TFT, TFTT}
    BpA::V1
    BmAfact::V2
    tmp::NT
    Δt::Float64
    α::Float64
    D::Matrix{Float64}
    D²::Matrix{Float64}
    fft::FT
    ifft::IFT
    tmpField::TFT
    tmpFTField::TFTT

    function StreamFunEquation(P::Int, L::Int, Lx::Real, Re::Real, Δt::Real)
        D  = chebdiff(P)
        y  = chebpoints(P)
        u₀ = 1 .- y.^2
        α = 2π/Lx

        # matrices for the time stepping
        B = [D*D - (l*α)^2*I for l = 0:L]
        A = [B[l+1]*B[l+1]/Re - 2*im*l*α*I - diagm(0=>u₀*im*l*α)*B[l+1] for l = 0:L]
        BpA = [B[l+1] .+ 0.5 .* Δt .* A[l+1] for l = 0:L]
        BmA = [B[l+1] .- 0.5 .* Δt .* A[l+1] for l = 0:L]
        BmAfact = [lu!( [ basis_vector(1, P+1)';
                          D[1, :]';
                          BmA[l+1][3:end-2, :];
                          D[end, :]';
                          basis_vector(P+1, P+1)' ] ) for l = 0:L]

        # temporaries
        tmp = (zeros(Complex{Float64}, P+1), zeros(Complex{Float64}, P+1))
        tmpField = ntuple(i->Field(P, L), 5)
        tmpFTField = ntuple(i->FTField(P, L), 5)

        # ffts
        fft, ifft = ForwardFFT!(tmpField[1]), InverseFFT!(tmpFTField[1])

        return new{P,
                   L,
                   typeof(BpA),
                   typeof(BmAfact),
                   typeof(tmp),
                   typeof(fft),
                   typeof(ifft),
                   typeof(tmpField),
                   typeof(tmpFTField)}(BpA, BmAfact, tmp, Δt, α,
                   D, D*D, fft, ifft, tmpField, tmpFTField)
    end
end


function (eq::StreamFunEquation)(t::Real, ψ̂::FTField, N̂::FTField)

    # # # aliases
    û, v̂, ω̂, dω̂dy, dω̂dx = eq.tmpFTField
    u, v, N, dωdy, dωdx = eq.tmpField

    # calculate velocities
    ddy!(û, ψ̂, eq)
    ddx!(v̂, ψ̂, eq); v̂ .*= -1

    # calculate vorticity
    laplacian!(ω̂, ψ̂, eq)

    # and its derivatives
    ddy!(dω̂dy, ω̂, eq)
    ddx!(dω̂dx, ω̂, eq)

    # transform
    eq.ifft(dωdy, dω̂dy)
    eq.ifft(dωdx, dω̂dx)
    eq.ifft(u, û)
    eq.ifft(v, v̂)

    # calc N
    N .= .- u .* dωdx .- v .* dωdy

    # and invert
    eq.fft(N̂, N)

    return N̂
end


function Flows.ImcA_mul!(eq::StreamFunEquation{P, L},
                          c::Real,
                          ψ::FTField{P, L},
                        out::FTField{P, L}) where {P, L}
    # check time step
    abs(c) == 0.5 * eq.Δt || throw("invalid time step size")

    # compute product
    tmp1, tmp2 = eq.tmp
    for l = 0:L
        _copycol!(tmp1, ψ.data, l+1)
        mul!(tmp2, eq.BpA[l+1], tmp1)
        _copycol!(out.data, tmp2, l+1)
    end

    return out
end

@inline function Flows.ImcA!(eq::StreamFunEquation{P, L},
                              c::Real,
                              ψ::FTField{P, L, T},
                            out::FTField{P, L, T}) where {P, L, T}
    # check time step
    abs(c) == 0.5 * eq.Δt || throw("invalid time step size")

    # solve systems
    tmp1, tmp2 = eq.tmp

    for l = 0:L
        _copycol!(tmp1, ψ.data, l+1)
        tmp1[1] = 0; tmp1[end]   = 0
        tmp1[2] = 0; tmp1[end-1] = 0
        ldiv!(eq.BmAfact[l+1], tmp1)
        _copycol!(out.data, tmp1, l+1)
    end

    return out
end


# utils to avoid allocation of views (get rid of this in julia 1.5)
# https://github.com/JuliaLang/julia/pull/34126
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

