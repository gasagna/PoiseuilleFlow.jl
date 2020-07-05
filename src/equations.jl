import LinearAlgebra: I, lu!, diagm, mul!, ldiv!
import Flows

export StreamFunEquation

struct StreamFunEquation{P, L, V1, V2, NT, FT, IFT, TFT, TFTT}
    BpA::V1
    BmAfact::V2
    tmp::NT
    Δt::Float64
    fft::FT
    ifft::IFT
    tmpPField::TFT
    tmpSField::TFTT

    function StreamFunEquation(grid::Grid, Re::Real, Δt::Real; 
                               flags::UInt32=FFTW.PATIENT, timelimit::Real=FFTW.NO_TIMELIMIT)
        # make mesh
        x, y = points(grid)

        # domain
        Lx, _ = domain(grid)

        # size of transforms
        P, L = size(grid, :spectral)

        # laminar flow and fundamental wavenumber
        u₀ = 1 .- y.^2
        α = 2π/Lx

        # extract actual DiffMatrices from the grid multipliers
        D1, D2, D4  = ntuple(i->grid.Ds[i].mat, 3)

        B = [D2 - (l*α)^2*I for l = 0:L]
        A = [(D4 + (l*α)^4*I - D2*((l*α)^2*I) - ((l*α)^2*I)*D2)/Re- 2*im*l*α*I - diagm(0=>u₀*im*l*α)*B[l+1] for l = 0:L]
        BpA = [B[l+1] + 0.5 * Δt * A[l+1] for l = 0:L]

        # matrices for the boundary value problems
        BmA = [B[l+1] - 0.5 * Δt * A[l+1] for l = 0:L]
        
        # add boundary conditions and factorise
        BmAfact = map(0:L) do l
            BmA_ = BmA[l+1]
            BmA_[1,     :] .= FDGrids.basis_vector(1, P)
            BmA_[2,     :] .= D1[1,   :]
            BmA_[end-1, :] .= D1[end, :]
            BmA_[end,   :] .= FDGrids.basis_vector(P, P)
            return lu!(BmA_)
        end

        # temporaries
        tmp = ntuple(i->zeros(Complex{Float64}, P), 2)
        tmpPField = ntuple(i->PhysicalField(grid), 5)
        tmpSField = ntuple(i->SpectralField(grid), 5)

        # ffts
         fft = ForwardFFT!(tmpPField[1], flags, timelimit)
        ifft = InverseFFT!(tmpSField[1], flags, timelimit)

        return new{P,
                   L,
                   typeof(BpA),
                   typeof(BmAfact),
                   typeof(tmp),
                   typeof(fft),
                   typeof(ifft),
                   typeof(tmpPField),
                   typeof(tmpSField)}(BpA, BmAfact, tmp, Δt, fft, 
                                      ifft, tmpPField, tmpSField)
    end
end


function (eq::StreamFunEquation)(t::Real, ψ̂::SpectralField, N̂::SpectralField)

    # # # aliases
    û, v̂, ω̂, dω̂dy, dω̂dx = eq.tmpSField
    u, v, N, dωdy, dωdx = eq.tmpPField

    # calculate velocities
    ddy!(û, ψ̂)
    ddx!(v̂, ψ̂); v̂ .*= -1

    # calculate vorticity
    laplacian!(ω̂, ψ̂)

    # and its derivatives
    ddy!(dω̂dy, ω̂)
    ddx!(dω̂dx, ω̂)

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