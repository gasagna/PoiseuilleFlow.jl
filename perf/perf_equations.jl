using PoiseuilleFlow
using FFTW
using Printf
using Flows
using LinearAlgebra

FFTW.set_num_threads(1)
LinearAlgebra.BLAS.set_num_threads(1)

P  = 101
Lx = 6π
α  = 2π / Lx
Re = 1000
Δt = 0.01;

for LD in 20:400
    # active number of waves
    L = down_dealias_size(LD)

    # create fields
    ψ = PhysicalField(P, LD, Lx, (x, y)->0.06*cos(3*α*x)*(1-y^2)^2 + 
                                         0.04*cos(5*α*x)*(1-y^2)^2 + 
                                         0.02*cos(6*α*x)*(1-y^2)^2)
    ψ̂ = SpectralField(P, L, LD)

    # inverse transform
    _ifft = InverseFFT!(ψ̂, FFTW.PATIENT, FFTW.NO_TIMELIMIT);
    _fft  = ForwardFFT!(ψ, FFTW.PATIENT, FFTW.NO_TIMELIMIT);

    # set initial condition
    _fft(ψ̂, ψ);

    # create equations
    eq = StreamFunEquation(P, L, LD, Lx, Re, Δt; flags=FFTW.PATIENT, timelimit=FFTW.NO_TIMELIMIT)

    # create flow operator
    ϕ = flow(eq, eq, CNRK2(ψ̂), TimeStepConstant(Δt));

    # march
    t = minimum([@elapsed ϕ(ψ̂, (0, Δt)) for i = 1:10])

    @printf "%10.2f %04d\n" 10^9 * t / ((P+1) * (2LD+2) * log(2LD+2)) LD
    flush(stdout)
end