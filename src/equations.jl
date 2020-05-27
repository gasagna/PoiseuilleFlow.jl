import HelmoltzSolvers: CoupledHelmoltzSolver, solve!, ChebCoeffs
import Flows
import FFTW

export StreamFunEquation

struct StreamFunEquation{P, L, TPF, TSF, S, FT, IFT, R}
    tmpPField::TPF
    tmpSField::TSF
       solver::S
           Re::Float64
          fft::FT
         ifft::IFT
         r_re::R
         r_im::R

    function StreamFunEquation( P::Int,   L::Int,
                               Lx::Real, Re::Real; flags::UInt32=FFTW.MEASURE)
        # helmoltz solver
        solver = CoupledHelmoltzSolver(P)

        # temporary storage for the solution of the Helmoltz problems
        r_re = ChebCoeffs(P)
        r_im = ChebCoeffs(P)

        # temporaries
        tmpPField = ntuple(i->PhysicalField(P, L, Lx), 5)
        tmpSField = ntuple(i->SpectralField(P, L, Lx), 5)

        # ffts
         fft = ForwardFFT!(tmpPField[1], flags)
        ifft = InverseFFT!(tmpSField[1], flags)

        return new{P,
                   L,
                   typeof(tmpPField),
                   typeof(tmpSField),
                   typeof(solver),
                   typeof(fft),
                   typeof(ifft),
                   typeof(r_re)}(tmpPField, tmpSField, solver, Re, fft, ifft, r_re, r_im)
    end
end


function (eq::StreamFunEquation)(t::Real, ψ̂::SpectralField, N̂::SpectralField)
    # the state is always the streamfunction because we can calculate the
    # velocity components from it directly, rather than as a solution of a BVP

    # aliases
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

    # add laminar flow velocity to û before transforming
    # these are the Chebyshev coefficients of the polynomial
    # u₀(y) = 1 - y^2 = c₀ T₀(y) + c₂ T₂(y)
    @inbounds û[0, 0] += 0.5
    @inbounds û[2, 0] -= 0.5

    # transform
    eq.ifft(dωdy, dω̂dy)
    eq.ifft(dωdx, dω̂dx)
    eq.ifft(u, û)
    eq.ifft(v, v̂)

    # calc N
    N .= .- u .* dωdx .- v .* (dωdy .- 2)

    # and invert
    eq.fft(N̂, N)

    return N̂
end


function Flows.ImcA_mul!(eq::StreamFunEquation{P, L},
                          c::Real,
                          ψ::SpectralField{P, L},
                        out::SpectralField{P, L}) where {P, L}

    # calculate vorticity from streamfunction and write to out
    ω = laplacian!(eq.tmpSField[1], ψ)

    # calculate laplacian of the vorticity
    laplacian!(out, ω)

    # calc actual term
    out .= 1 .- c .* out ./ eq.Re

    return out
end

"""
    Solve (I - dt/2/Re[D² - l²α²])ωₗ = rₗ
                       (D² - l²α²)ψₗ = ωₗ
    with
                    ψₗ(±1) = ψₗ'(±1) = 0
    for all l ∈ [0, P].

    The input argument `r` is calculated on the vorticity equation
"""
@inline function Flows.ImcA!(eq::StreamFunEquation{P, L},
                              c::Real,
                              r::SpectralField{P, L, Lx, T},
                              ψ::SpectralField{P, L, Lx, T}) where {P, L, Lx, T}
    # fundamental wavenumber
    α = 2π/Lx

    for l = 0:L
        # coefficients of the Helmoltz problem
        θ₀ = -c/eq.Re # c is usually Δt/2 for the forward problem (check adjoint)
        θ₁ = -(1 + (l*α)^2 / eq.Re)
        θ₂ = 1
        θ₃ = (l*α)^2

        # copy column of r, solve, then copy r back to the l-th column of ψ
        _copycol!(eq.r_re, r, l, real)
        _copycol!(eq.r_im, r, l, imag)
        solve!(eq.solver, (θ₀, θ₁, θ₂, θ₃), eq.r_re)
        solve!(eq.solver, (θ₀, θ₁, θ₂, θ₃), eq.r_im)
        _copycol!(ψ, eq.r_re, eq.r_im, l)
    end

    return ψ
end

# utils to avoid allocation of views (get rid of this in julia 1.5)
# https://github.com/JuliaLang/julia/pull/34126
function _copycol!(r::ChebCoeffs{T, P}, ψ::SpectralField{P}, c::Int, fun::F) where {T, P, F}
    @inbounds @simd for p = 0:P
        r[p] = fun(ψ[p, c])
    end
    return nothing
end

function _copycol!(ψ::SpectralField{P}, r_re::ChebCoeffs{T, P}, r_im::ChebCoeffs{T, P}, c::Int) where {T, P}
    @inbounds @simd for p = 0:P
        ψ[p, c] = r_re[p] + im * r_im[p]
    end
    return nothing
end

