module PoiseuilleFlow

using LinearAlgebra
using FDGrids

include("physicalfield.jl")
include("norms.jl")
include("spectralfield.jl")
include("multiplier.jl")
include("ffts.jl")
include("chebutils.jl")
include("equations.jl")
include("operators.jl")

end