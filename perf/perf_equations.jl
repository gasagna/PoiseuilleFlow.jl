using PoiseuilleFlow
using FFTW
using Printf
using Flows
using LinearAlgebra

FFTW.set_num_threads(1)
LinearAlgebra.BLAS.set_num_threads(1)

P  = 101
Lx = 6π
η  = 0.3
α  = 2π / Lx
Re = 1000
Δt = 0.01;
width = 3

for LD in 250:255
    # active number of waves
    L = down_dealias_size(LD)

    # make grid
    g = Grid(P, L, LD, 1; η=0.5, width=5)

    # create fields
    ψ = PhysicalField(g, (x, y)->0.06*cos(3*α*x)*(1-y^2)^2 + 
                                 0.04*cos(5*α*x)*(1-y^2)^2 + 
                                 0.02*cos(6*α*x)*(1-y^2)^2)
    ψ̂ = SpectralField(g)

    # inverse transform
    _ifft = InverseFFT!(ψ̂, FFTW.PATIENT, FFTW.NO_TIMELIMIT);
    _fft  = ForwardFFT!(ψ, FFTW.PATIENT, FFTW.NO_TIMELIMIT);

    # set initial condition
    _fft(ψ̂, ψ);

    # create equations
    eq = StreamFunEquation(g, Re, Δt; flags=FFTW.PATIENT, timelimit=FFTW.NO_TIMELIMIT)

    # create flow operator
    ϕ = flow(eq, eq, CNRK2(ψ̂), TimeStepConstant(Δt));

    # march
    t = minimum([@elapsed ϕ(ψ̂, (0, Δt)) for i = 1:30])

    @printf "%10.2f %04d\n" 10^9 * t / ((P+1) * (2LD+2) * log(2LD+2)) LD
    flush(stdout)
end