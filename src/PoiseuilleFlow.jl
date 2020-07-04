module PoiseuilleFlow

using LinearAlgebra
using FDGrids
using DelimitedFiles

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("ffts.jl")
include("operators.jl")
include("multiplier.jl")
include("equations.jl")

# include("norms.jl")

end