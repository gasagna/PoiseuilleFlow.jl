import LinearAlgebra: I, lu!, diagm, mul!, ldiv!
import Flows

export StreamFunEquation

struct StreamFunEquation{P, L, V1, V2, NT, FT, IFT, TFT, TFTT, M}
    BpA::V1
    BmAfact::V2
    tmp::NT
    Δt::Float64
    α::Float64
    D::M
    D²::M
    fft::FT
    ifft::IFT
    tmpPField::TFT
    tmpSField::TFTT

    function StreamFunEquation(P::Int,   L::Int,  LD::Int, Lx::Real, 
                               η::Real, Re::Real, Δt::Real; flags::UInt32=FFTW.EXHAUSTIVE, 
                                                        timelimit::Real=FFTW.NO_TIMELIMIT, 
                                                            width::Int=7)
        # make mesh
        y  = gridpoints(P+1, -1, 1, η)

        # laminar flow and fundamental wavenumber
        u₀ = 1 .- y.^2
        α = 2π/Lx

        # matrices for the time stepping
        D  = DiffMatrix(y, width,   1)
        D2 = DiffMatrix(y, width,   2)
        D4 = DiffMatrix(y, width+2, 4)

        B = [D2 - (l*α)^2*I for l = 0:L]
        A = [(D4 + (l*α)^4*I - D2*((l*α)^2*I) - ((l*α)^2*I)*D2)/Re- 2*im*l*α*I - diagm(0=>u₀*im*l*α)*B[l+1] for l = 0:L]
        BpA = [B[l+1] + 0.5 * Δt * A[l+1] for l = 0:L]

        # matrices for the boundary value problems
        BmA = [B[l+1] - 0.5 * Δt * A[l+1] for l = 0:L]
        
        # add boundary conditions and factorise
        BmAfact = map(0:L) do l
            BmA_ = BmA[l+1]
            BmA_[1,     :] .= basis_vector(1,  P+1)
            BmA_[2,     :] .= D[1,   :]
            BmA_[end-1, :] .= D[end, :]
            BmA_[end,   :] .= basis_vector(P+1, P+1)
            return lu!(BmA_)
        end

        # temporaries
        tmp = ntuple(i->zeros(Complex{Float64}, P+1), 2)
        tmpPField = ntuple(i->PhysicalField(P, LD, η, Lx), 5)
        tmpSField = ntuple(i->SpectralField(P, L, LD), 5)

        # ffts
         fft = ForwardFFT!(tmpPField[1], flags, timelimit)
        ifft = InverseFFT!(tmpSField[1], flags, timelimit)

        # fast multpliers
        MD  = Multiplier(D)
        MDD = Multiplier(D2)

        return new{P,
                   L,
                   typeof(BpA),
                   typeof(BmAfact),
                   typeof(tmp),
                   typeof(fft),
                   typeof(ifft),
                   typeof(tmpPField),
                   typeof(tmpSField),
                   typeof(MD)}(BpA, BmAfact, tmp, Δt, α,
                   MD, MDD, fft, ifft, tmpPField, tmpSField)
    end
end


function (eq::StreamFunEquation)(t::Real, ψ̂::SpectralField, N̂::SpectralField)

    # # # aliases
    û, v̂, ω̂, dω̂dy, dω̂dx = eq.tmpSField
    u, v, N, dωdy, dωdx = eq.tmpPField

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
                          ψ::SpectralField{P, L},
                        out::SpectralField{P, L}) where {P, L}
    # check time step
    abs(c) == 0.5 * eq.Δt || throw("invalid time step size")

    # compute product
    tmp1, tmp2 = eq.tmp
    @inbounds for l = 0:L
        _copycol!(tmp1, ψ.data, l+1)
        mul!(tmp2, eq.BpA[l+1], tmp1)
        _copycol!(out.data, tmp2, l+1)
    end

    return out
end

@inline function Flows.ImcA!(eq::StreamFunEquation{P, L},
                              c::Real,
                              ψ::SpectralField{P, L},
                            out::SpectralField{P, L}) where {P, L}
    # check time step
    abs(c) == 0.5 * eq.Δt || throw("invalid time step size")

    # solve systems
    tmp1, tmp2 = eq.tmp

    @inbounds for l = 0:L
        _copycol!(tmp1, ψ.data, l+1)
        tmp1[1] = 0; tmp1[end]   = 0
        tmp1[2] = 0; tmp1[end-1] = 0
        ldiv!(eq.BmAfact[l+1], tmp1)
        _copycol!(out.data, tmp1, l+1)
    end

    return out
end