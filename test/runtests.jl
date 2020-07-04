using BenchmarkTools
using PoiseuilleFlow
using LinearAlgebra
using Flows
using FFTW
using Test

include("test_grid.jl")
include("test_physicalfield.jl")
include("test_spectralfield.jl")
include("test_ffts.jl")
include("test_operators.jl")
include("test_equations.jl")
